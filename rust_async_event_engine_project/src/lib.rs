//! # Event Engine RS (Fixed & Optimized)
//!
//! 修复了 Arc::make_mut 在 Py<T> 上无法使用标准 Clone 的问题。
//! 实现了手动的 Copy-On-Write 机制，确保在持有 GIL 的情况下进行克隆。
//! **修复了 stop 时因解释器关闭导致的 Panic 问题 (使用 FFI 检查)。**
//! **修复了编译错误：移除不存在的 PyMethod，使用稳健的 eq 比较策略。**

use chrono::Local;
use log::{debug, error};
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
// [修复] 移除了 use pyo3::types::PyMethod; 因为该类型未公开
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tokio::runtime::Builder;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::time::{interval, Duration};

// ============================================================================
// 常量定义
// ============================================================================
pub const EVENT_TIMER: &str = "eTimer.";
/// 批处理大小
const BATCH_SIZE: usize = 64;

// ============================================================================
// 数据结构定义
// ============================================================================
type HandlerList = Arc<Vec<Py<PyAny>>>;
type HandlerStorage = Arc<RwLock<HashMap<String, HandlerList>>>;
type GeneralHandlerStorage = Arc<RwLock<HandlerList>>;

// ============================================================================
// 辅助函数：Handler 比较
// ============================================================================
/// 判定两个 Handler 是否相同
/// 策略：
/// 仅使用 Python 原生 eq (适用于 Bound Method 和自定义对象)
/// 移除了指针 (as_ptr) 快速对比，完全依赖 Python 解释器的 __eq__ 逻辑
fn is_same_handler(target: &Bound<PyAny>, current: &Bound<PyAny>) -> bool {
    // [修改] 直接调用 eq，不进行指针预检查
    target.eq(current).unwrap_or(false)
}

// ============================================================================
// Event 类
// ============================================================================
#[pyclass(name = "Event")]
pub struct Event {
    #[pyo3(get, set)]
    pub type_: String,
    #[pyo3(get, set)]
    pub data: Option<Py<PyAny>>,
}

struct InternalEvent {
    type_: String,
    data: Option<Py<PyAny>>,
}

impl Clone for Event {
    fn clone(&self) -> Self {
        Python::attach(|py| Event {
            type_: self.type_.clone(),
            data: self.data.as_ref().map(|d| d.clone_ref(py)),
        })
    }
}

#[pymethods]
impl Event {
    #[new]
    #[pyo3(signature = (type_, data=None))]
    fn new(type_: String, data: Option<Py<PyAny>>) -> PyResult<Self> {
        if type_.is_empty() {
            return Err(PyValueError::new_err("Event type_ cannot be empty"));
        }
        Ok(Event { type_, data })
    }

    fn __repr__(&self) -> String {
        format!("Event(type_='{}', data={:?})", self.type_, self.data.is_some())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// EventEngine 类
// ============================================================================
#[pyclass(name = "EventEngine")]
pub struct EventEngine {
    #[pyo3(get)]
    interval: u64,
    #[pyo3(get, set)]
    channel: String,
    active: Arc<AtomicBool>,
    handlers: HandlerStorage,
    general_handlers: GeneralHandlerStorage,
    sender: Option<UnboundedSender<InternalEvent>>,
    thread_handle: Option<JoinHandle<()>>,
}

#[pymethods]
impl EventEngine {
    #[new]
    #[pyo3(signature = (interval=1))]
    fn new(interval: u64) -> PyResult<Self> {
        let actual_interval = if interval == 0 { 1 } else { interval };
        Ok(EventEngine {
            interval: actual_interval,
            channel: String::new(),
            active: Arc::new(AtomicBool::new(false)),
            handlers: Arc::new(RwLock::new(HashMap::new())),
            general_handlers: Arc::new(RwLock::new(Arc::new(Vec::new()))),
            sender: None,
            thread_handle: None,
        })
    }

    fn start(&mut self) -> PyResult<()> {
        if self.active.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.active.store(true, Ordering::SeqCst);

        let (sender, receiver) = mpsc::unbounded_channel();
        self.sender = Some(sender.clone());

        let active = self.active.clone();
        let handlers = self.handlers.clone();
        let general_handlers = self.general_handlers.clone();
        let interval_secs = self.interval;

        let handle = thread::spawn(move || {
            let runtime = match Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    error!("Failed to create runtime: {}", e);
                    return;
                }
            };

            runtime.block_on(async {
                run_event_loop(
                    active,
                    handlers,
                    general_handlers,
                    receiver,
                    sender,
                    interval_secs,
                ).await;
            });
        });
        self.thread_handle = Some(handle);
        Ok(())
    }

    fn stop(&mut self) -> PyResult<()> {
        if !self.active.swap(false, Ordering::SeqCst) {
            return Ok(());
        }
        self.sender = None;
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        Ok(())
    }

    fn is_loop_running(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    fn put(&self, event: &Event) -> PyResult<()> {
        self.event_to_queue(event)
    }

    fn event_to_queue(&self, event: &Event) -> PyResult<()> {
        if !self.is_loop_running() {
            return Ok(());
        }
        
        let internal = Python::attach(|py| InternalEvent {
            type_: event.type_.clone(),
            data: event.data.as_ref().map(|d| d.clone_ref(py)),
        });

        if let Some(ref sender) = self.sender {
            if let Err(e) = sender.send(internal) {
                error!("Failed to queue event: {}", e);
            }
        }
        Ok(())
    }

    // 使用 is_same_handler
    fn register(&self, type_: String, handler: Py<PyAny>) -> PyResult<()> {
        if type_.is_empty() {
            return Err(PyValueError::new_err("Type empty"));
        }
        Python::attach(|py| {
            let mut guard = self.handlers.write();
            let list_arc = guard.entry(type_).or_insert_with(|| Arc::new(Vec::new()));
            
            // Manual COW
            let list = if let Some(list) = Arc::get_mut(list_arc) {
                list
            } else {
                let new_vec: Vec<Py<PyAny>> = list_arc.iter().map(|h| h.clone_ref(py)).collect();
                *list_arc = Arc::new(new_vec);
                Arc::get_mut(list_arc).unwrap()
            };

            let bound_handler = handler.bind(py);
            // 使用优化后的比较逻辑
            let exists = list.iter().any(|h| {
                is_same_handler(bound_handler, h.bind(py))
            });

            if !exists {
                list.push(handler);
            }
        });
        Ok(())
    }

    // 使用 is_same_handler
    fn unregister(&self, type_: String, handler: Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| {
            let mut guard = self.handlers.write();
            if let Some(list_arc) = guard.get_mut(&type_) {
                // Manual COW
                let list = if let Some(list) = Arc::get_mut(list_arc) {
                    list
                } else {
                    let new_vec: Vec<Py<PyAny>> = list_arc.iter().map(|h| h.clone_ref(py)).collect();
                    *list_arc = Arc::new(new_vec);
                    Arc::get_mut(list_arc).unwrap()
                };

                let bound_handler = handler.bind(py);
                // 使用优化后的比较逻辑保留不相同的
                list.retain(|h| {
                    !is_same_handler(bound_handler, h.bind(py))
                });

                if list.is_empty() {
                    guard.remove(&type_);
                }
            }
        });
        Ok(())
    }

    // 使用 is_same_handler
    fn register_general(&self, handler: Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| {
            let mut guard = self.general_handlers.write();
            let list_arc = &mut *guard;
            
            // Manual COW
            let list = if let Some(list) = Arc::get_mut(list_arc) {
                list
            } else {
                let new_vec: Vec<Py<PyAny>> = list_arc.iter().map(|h| h.clone_ref(py)).collect();
                *list_arc = Arc::new(new_vec);
                Arc::get_mut(list_arc).unwrap()
            };

            let bound_handler = handler.bind(py);
            // 使用优化后的比较逻辑
            let exists = list.iter().any(|h| {
                is_same_handler(bound_handler, h.bind(py))
            });

            if !exists {
                list.push(handler);
            }
        });
        Ok(())
    }

    // 使用 is_same_handler
    fn unregister_general(&self, handler: Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| {
            let mut guard = self.general_handlers.write();
            let list_arc = &mut *guard;
            
            // Manual COW
            let list = if let Some(list) = Arc::get_mut(list_arc) {
                list
            } else {
                let new_vec: Vec<Py<PyAny>> = list_arc.iter().map(|h| h.clone_ref(py)).collect();
                *list_arc = Arc::new(new_vec);
                Arc::get_mut(list_arc).unwrap()
            };

            let bound_handler = handler.bind(py);
            // 使用优化后的比较逻辑
            list.retain(|h| {
                !is_same_handler(bound_handler, h.bind(py))
            });
        });
        Ok(())
    }

    fn process(&self, py: Python<'_>, event: &Event) -> PyResult<()> {
        let specific_handlers = {
            let guard = self.handlers.read();
            guard.get(&event.type_).cloned()
        };
        let general_handlers = {
            self.general_handlers.read().clone()
        };

        if let Some(handlers) = specific_handlers {
            for handler in handlers.iter() {
                if let Err(e) = handler.call1(py, (event.clone(),)) {
                    e.print(py);
                }
            }
        }

        for handler in general_handlers.iter() {
            if let Err(e) = handler.call1(py, (event.clone(),)) {
                e.print(py);
            }
        }
        Ok(())
    }
}

// ============================================================================
// 异步事件循环
// ============================================================================
async fn run_event_loop(
    active: Arc<AtomicBool>,
    handlers: HandlerStorage,
    general_handlers: GeneralHandlerStorage,
    mut receiver: UnboundedReceiver<InternalEvent>,
    sender: UnboundedSender<InternalEvent>,
    interval_secs: u64,
) {
    let timer_active = active.clone();
    let timer_sender = sender.clone();
    
    tokio::spawn(async move {
        run_timer(timer_active, timer_sender, interval_secs).await;
    });

    let mut event_buffer = Vec::with_capacity(BATCH_SIZE);

    while active.load(Ordering::Relaxed) {
        let first_event = tokio::select! {
            res = receiver.recv() => match res {
                Some(e) => e,
                None => break,
            },
             _= tokio::time::sleep(Duration::from_millis(100)) => {
                continue;
            }
        };

        event_buffer.push(first_event);
        while event_buffer.len() < BATCH_SIZE {
            match receiver.try_recv() {
                Ok(e) => event_buffer.push(e),
                Err(_) => break,
            }
        }

        if !event_buffer.is_empty() {
            // [修复] 使用 FFI 检查解释器状态
            if unsafe { ffi::Py_IsInitialized() } == 0 {
                debug!("Python interpreter shutdown detected, stopping loop.");
                break;
            }

            Python::attach(|py| {
                for internal_event in event_buffer.drain(..) {
                    let specific_handlers_opt = {
                        let guard = handlers.read();
                        guard.get(&internal_event.type_).cloned()
                    };
                    let general_handlers_arc = {
                        general_handlers.read().clone()
                    };

                    let py_event = Event {
                        type_: internal_event.type_,
                        data: internal_event.data,
                    };
                    
                    let py_event_obj = match Py::new(py, py_event) {
                        Ok(obj) => obj,
                        Err(e) => {
                            e.print(py);
                            continue;
                        }
                    };

                    if let Some(handlers) = specific_handlers_opt {
                        for handler in handlers.iter() {
                            if let Err(e) = handler.call1(py, (py_event_obj.clone_ref(py),)) {
                                error!("Handler failed");
                                e.print(py);
                            }
                        }
                    }

                    if !general_handlers_arc.is_empty() {
                        for handler in general_handlers_arc.iter() {
                            if let Err(e) = handler.call1(py, (py_event_obj.clone_ref(py),)) {
                                error!("General handler failed");
                                e.print(py);
                            }
                        }
                    }
                }
            });
        }
    }
}

async fn run_timer(
    active: Arc<AtomicBool>,
    sender: UnboundedSender<InternalEvent>,
    interval_secs: u64,
) {
    let mut timer = interval(Duration::from_secs(interval_secs));
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    while active.load(Ordering::Relaxed) {
        timer.tick().await;
        if !active.load(Ordering::Relaxed) { break; }

        // [修复] 使用 FFI 检查解释器状态
        if unsafe { ffi::Py_IsInitialized() } == 0 {
            debug!("Python interpreter shutdown detected, stopping timer.");
            break;
        }

        let now = Local::now();
        let success = Python::attach(|py| {
            let datetime_str = now.format("%Y-%m-%d %H:%M:%S%.3f").to_string();
            match datetime_str.into_pyobject(py) {
                Ok(pystr) => {
                    let event = InternalEvent {
                        type_: EVENT_TIMER.to_string(),
                        data: Some(pystr.unbind().into_any()),
                    };
                    sender.send(event).is_ok()
                },
                Err(_) => false
            }
        });

        if !success && active.load(Ordering::Relaxed) {
             debug!("Timer event send failed or channel closed");
        }
    }
}

// ============================================================================
// Python 模块定义
// ============================================================================
#[pymodule]
fn rust_async_event_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Event>()?;
    m.add_class::<EventEngine>()?;
    m.add("EVENT_TIMER", EVENT_TIMER)?;
    Ok(())
}
