# Body Cam - 身体动作捕捉系统

一个基于 AI 的实时身体动作识别和分析系统，使用 MediaPipe 进行姿态检测，整合 OpenAI API 进行智能分析。

## 功能特性

- 🎥 **实时视频处理** - 支持摄像头或视频文件输入
- 🧍 **人体姿态识别** - 使用 MediaPipe 检测 33 个关键点
- 🤖 **AI 智能分析** - 集成 OpenAI API 进行动作识别和反馈
- 📊 **数据输出** - 支持 JSON/CSV 格式导出
- 🐳 **Docker 支持** - 一键部署到容器环境

## 快速开始

### 前置要求

- Python 3.8+
- OpenCV
- MediaPipe
- OpenAI API Key

### 安装

**方式一：本地安装**

```bash
# 克隆仓库
git clone https://github.com/allul423-netizen/body_cam.git
cd body_cam

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

**方式二：Docker 部署**

```bash
docker-compose up -d
```

### 配置

编辑 `config.yml` 文件配置参数：

```yaml
camera:
  device_id: 0
  fps: 30
  frame_width: 1280
  frame_height: 720

pose_detection:
  model_complexity: 1
  min_detection_confidence: 0.7

openai:
  api_key: "your-api-key"
  model: "gpt-4"
```

### 使用

```bash
python main.py
```

## 项目结构

```
body_cam/
├── main.py                 # 主程序入口
├── config.yml              # 配置文件
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 镜像配置
├── docker-compose.yml      # Docker 编排
├── .gitignore              # Git 忽略规则
└── README.md               # 项目说明
```

## 文件说明

- **config.yml** - 摄像头、模型、API 等配置参数
- **requirements.txt** - 所有 Python 依赖包
- **Dockerfile** - 容器化部署配置
- **docker-compose.yml** - 多容器编排定义

## API 密钥配置

设置 OpenAI API 密钥：

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key"

# Windows 命令行
set OPENAI_API_KEY=your-api-key

# Linux/Mac
export OPENAI_API_KEY="your-api-key"
```

## 输出示例

系统识别到的动作和分析结果：

```json
{
  "timestamp": "2026-02-18T10:30:45",
  "detected_action": "跳跃",
  "confidence": 0.92,
  "analysis": "用户执行了标准的双脚跳跃动作，姿态良好"
}
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

- GitHub: [@allul423-netizen](https://github.com/allul423-netizen)
- Email: your-email@example.com

---

**最后更新**: 2026年2月18日