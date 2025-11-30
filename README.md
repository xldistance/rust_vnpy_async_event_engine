## 前置要求

### 安装 Rust

```bash
# Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Windows
# 下载并运行 rustup-init.exe
# https://rustup.rs/
```

安装完成后，重启终端并验证安装：

```bash
rustc --version
cargo --version
```
## 安装步骤

```bash
pip install maturin
```
**生产构建：**
```bash
maturin build --release
pip install target/wheels/*.whl
```
## 使用示例
修改vnpy\event\__init__.py如下:
```
#from .engine import EVENT_TIMER, Event, EventEngine
from rust_async_event_engine import EVENT_TIMER, Event, EventEngine

```
