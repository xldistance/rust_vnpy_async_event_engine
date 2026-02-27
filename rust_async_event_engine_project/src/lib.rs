//! # Event Engine RS (重构版)
//!
//! 基于 PyO3 + Tokio 的高性能 Python 事件引擎。
//! 
//! ## 重构改进
//! - 提取通用 COW 操作，消除四处重复的手动 COW 代码
//! - 恢复指针快速路径 + Python __eq__ 的双层比较策略
//! - 统一 handler 的增删逻辑为泛型辅助函数
//! - 优化事件循环中的 GIL 获取粒度

use chrono::Local;
use log::{debug, error};
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tokio::runtime::Builder;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::time::{interval, Duration};

// ============================================================================
// 常量
// ============================================================================

pub const EVENT_TIMER: &str = "eTimer.";
const BATCH_SIZE: usize = 64;

// ============================================================================
// 类型别名
// ============================================================================

type HandlerList = Arc<Vec<Py<PyAny>>>;
type HandlerStorage = Arc<RwLock<HashMap<String, HandlerList>>>;
type GeneralHandlerStorage = Arc<RwLock<HandlerList>>;

// ============================================================================
// COW Handler 列表操作 (消除重复)
// ============================================================================

/// 对 `Arc<Vec<Py<PyAny>>>` 执行 Copy-On-Write，返回可变引用。
/// 当 Arc 引用计数 > 1 时，在持有 GIL 的情况下克隆整个 Vec。
fn cow_handler_list<'a>(list_arc: &'a mut HandlerList, py: Python<'_>) -> &'a mut Vec<Py<PyAny>> {
    if Arc::get_mut(list_arc).is_none() {
        let cloned: Vec<Py<PyAny>> = list_arc.iter().map(|h| h.clone_ref(py)).collect();
        *list_arc = Arc::new(cloned);
    }
    Arc::get_mut(list_arc).expect("COW: Arc should be unique after clone")
}

/// 向 handler 列表中添加 handler（去重）。
fn add_handler(list_arc: &mut HandlerList, handler: Py<PyAny>, py: Python<'_>) {
    let bound = handler.bind(py);
    let exists = list_arc.iter().any(|h| is_same_handler(bound, h.bind(py)));
    if exists {
        return;
    }
    let list = cow_handler_list(list_arc, py);
    list.push(handler);
}

/// 从 handler 列表中移除 handler。
fn remove_handler(list_arc: &mut HandlerList, handler: &Py<PyAny>, py: Python<'_>) {
    let bound = handler.bind(py);
    let has_match = list_arc.iter().any(|h| is_same_handler(bound, h.bind(py)));
    if !has_match {
        return;
    }
    let list = cow_handler_list(list_arc, py);
    list.retain(|h| !is_same_handler(bound, h.bind(py)));
}

// ============================================================================
// Handler 比较
// ============================================================================

/// 判定两个 handler 是否相同。
///
/// 策略：先用指针快速判等（覆盖同一对象的常见场景），
/// 再回退到 Python `__eq__`（覆盖 bound method 等场景）。
fn is_same_handler(a: &Bound<PyAny>, b: &Bound<PyAny>) -> bool {
    if a.as_ptr() == b.as_ptr() {
        return true;
    }
    a.eq(b).unwrap_or(false)
}

// ============================================================================
// Python 解释器存活检查
// ============================================================================

/// 检查 Python 解释器是否仍在运行（用于后台线程安全退出）。
#[inline]
fn is_python_alive() -> bool {
    unsafe { ffi::Py_IsInitialized() != 0 }
}

// ============================================================================
// InternalEvent
// ============================================================================

struct InternalEvent {
    type_: String,
    data: Option<Py<PyAny>>,
}

// ============================================================================
// Event
// ============================================================================

#[pyclass(name = "Event", from_py_object)]
pub struct Event {
    #[pyo3(get, set)]
    pub type_: String,
    #[pyo3(get, set)]
    pub data: Option<Py<PyAny>>,
}

impl Event {
    /// 在已持有 GIL 的上下文中克隆。
    fn clone_with_gil(&self, py: Python<'_>) -> Self {
        Event {
            type_: self.type_.clone(),
            data: self.data.as_ref().map(|d| d.clone_ref(py)),
        }
    }

    /// 转换为 InternalEvent（零拷贝移动语义，仅克隆 Py 引用）。
    fn to_internal(&self, py: Python<'_>) -> InternalEvent {
        InternalEvent {
            type_: self.type_.clone(),
            data: self.data.as_ref().map(|d| d.clone_ref(py)),
        }
    }
}

impl Clone for Event {
    fn clone(&self) -> Self {
        Python::attach(|py| self.clone_with_gil(py))
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
        format!(
            "Event(type_='{}', data={})",
            self.type_,
            if self.data.is_some() { "Some(...)" } else { "None" }
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// EventEngine
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
        Ok(EventEngine {
            interval: interval.max(1),
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

        let ctx = LoopContext {
            active: self.active.clone(),
            handlers: self.handlers.clone(),
            general_handlers: self.general_handlers.clone(),
            interval_secs: self.interval,
        };

        let handle = thread::spawn(move || {
            let runtime = match Builder::new_current_thread().enable_all().build() {
                Ok(rt) => rt,
                Err(e) => {
                    error!("Failed to create tokio runtime: {}", e);
                    return;
                }
            };
            runtime.block_on(run_event_loop(ctx, receiver, sender));
        });

        self.thread_handle = Some(handle);
        Ok(())
    }

    fn stop(&mut self) -> PyResult<()> {
        if !self.active.swap(false, Ordering::SeqCst) {
            return Ok(());
        }
        // 丢弃 sender 使 receiver 端收到 None，触发循环退出
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
        if !self.is_loop_running() {
            return Ok(());
        }
        let internal = Python::attach(|py| event.to_internal(py));
        if let Some(ref sender) = self.sender {
            if let Err(e) = sender.send(internal) {
                error!("Failed to queue event: {}", e);
            }
        }
        Ok(())
    }

    fn register(&self, type_: String, handler: Py<PyAny>) -> PyResult<()> {
        if type_.is_empty() {
            return Err(PyValueError::new_err("Type cannot be empty"));
        }
        Python::attach(|py| {
            let mut guard = self.handlers.write();
            let list_arc = guard.entry(type_).or_insert_with(|| Arc::new(Vec::new()));
            add_handler(list_arc, handler, py);
        });
        Ok(())
    }

    fn unregister(&self, type_: String, handler: Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| {
            let mut guard = self.handlers.write();
            if let Some(list_arc) = guard.get_mut(&type_) {
                remove_handler(list_arc, &handler, py);
                if list_arc.is_empty() {
                    guard.remove(&type_);
                }
            }
        });
        Ok(())
    }

    fn register_general(&self, handler: Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| {
            let guard = &mut *self.general_handlers.write();
            add_handler(guard, handler, py);
        });
        Ok(())
    }

    fn unregister_general(&self, handler: Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| {
            let guard = &mut *self.general_handlers.write();
            remove_handler(guard, &handler, py);
        });
        Ok(())
    }

    fn process(&self, py: Python<'_>, event: &Event) -> PyResult<()> {
        let specific = {
            let guard = self.handlers.read();
            guard.get(&event.type_).cloned()
        };
        let general = self.general_handlers.read().clone();

        let py_event = event.clone_with_gil(py);
        dispatch_to_handlers(py, &py_event, specific.as_ref().map(|v| v.as_slice()), &general);
        Ok(())
    }
}

// ============================================================================
// Handler 分发 (提取公共逻辑)
// ============================================================================

/// 将事件分发给特定 handler 列表和通用 handler 列表。
fn dispatch_to_handlers(
    py: Python<'_>,
    event: &Event,
    specific: Option<&[Py<PyAny>]>,
    general: &[Py<PyAny>],
) {
    let py_obj = match Py::new(py, event.clone_with_gil(py)) {
        Ok(obj) => obj,
        Err(e) => {
            e.print(py);
            return;
        }
    };

    if let Some(handlers) = specific {
        for handler in handlers {
            if let Err(e) = handler.call1(py, (py_obj.clone_ref(py),)) {
                error!("Specific handler error");
                e.print(py);
            }
        }
    }

    for handler in general {
        if let Err(e) = handler.call1(py, (py_obj.clone_ref(py),)) {
            error!("General handler error");
            e.print(py);
        }
    }
}

// ============================================================================
// 异步事件循环
// ============================================================================

/// 事件循环所需的共享上下文，减少函数参数数量。
struct LoopContext {
    active: Arc<AtomicBool>,
    handlers: HandlerStorage,
    general_handlers: GeneralHandlerStorage,
    interval_secs: u64,
}

async fn run_event_loop(
    ctx: LoopContext,
    mut receiver: UnboundedReceiver<InternalEvent>,
    sender: UnboundedSender<InternalEvent>,
) {
    // 启动定时器任务
    let timer_active = ctx.active.clone();
    let timer_sender = sender.clone();
    let interval_secs = ctx.interval_secs;
    tokio::spawn(async move {
        run_timer(timer_active, timer_sender, interval_secs).await;
    });

    let mut buf = Vec::with_capacity(BATCH_SIZE);

    while ctx.active.load(Ordering::Relaxed) {
        // 等待第一个事件或超时
        let first = tokio::select! {
            res = receiver.recv() => match res {
                Some(e) => e,
                None => break,
            },
            _ = tokio::time::sleep(Duration::from_millis(100)) => continue,
        };

        // 批量收集
        buf.push(first);
        while buf.len() < BATCH_SIZE {
            match receiver.try_recv() {
                Ok(e) => buf.push(e),
                Err(_) => break,
            }
        }

        if !is_python_alive() {
            debug!("Python interpreter shut down, exiting event loop.");
            break;
        }

        Python::attach(|py| {
            for ev in buf.drain(..) {
                let specific = {
                    let guard = ctx.handlers.read();
                    guard.get(&ev.type_).cloned()
                };
                let general = ctx.general_handlers.read().clone();

                let event = Event {
                    type_: ev.type_,
                    data: ev.data,
                };
                dispatch_to_handlers(py, &event, specific.as_ref().map(|v| v.as_slice()), &general);
            }
        });
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
        if !active.load(Ordering::Relaxed) {
            break;
        }
        if !is_python_alive() {
            debug!("Python interpreter shut down, exiting timer.");
            break;
        }

        let now = Local::now();
        let ok = Python::attach(|py| {
            let ts = now.format("%Y-%m-%d %H:%M:%S%.3f").to_string();
            match ts.into_pyobject(py) {
                Ok(pystr) => sender
                    .send(InternalEvent {
                        type_: EVENT_TIMER.to_string(),
                        data: Some(pystr.unbind().into_any()),
                    })
                    .is_ok(),
                Err(_) => false,
            }
        });

        if !ok && active.load(Ordering::Relaxed) {
            debug!("Timer event send failed or channel closed.");
        }
    }
}

// ============================================================================
// Python 模块
// ============================================================================

#[pymodule]
fn rust_async_event_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Event>()?;
    m.add_class::<EventEngine>()?;
    m.add("EVENT_TIMER", EVENT_TIMER)?;
    Ok(())
}
