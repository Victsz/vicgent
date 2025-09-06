```
ProjectRoot/
├── src/
│   └── pkgname/
│       ├── __init__.py
│       ├── interface/          # 接口层
│       │   ├── __init__.py
│       │   └── hello.py
│       ├── core/               # 核心逻辑层
│       │   ├── __init__.py
│       │   └── hello.py
│       └── util/               # 工具层
│           ├── __init__.py
│           └── hello.py
├── main.py                     # 主入口
├── pyproject.toml              # 项目配置
└── README.md # 留空
```    