数据下载前往 kaggle [CT-Images](https://www.kaggle.com/datasets/seifelmedany/ct-images) 或者使用终端下载：

```bash
#!/bin/bash
kaggle datasets download seifelmedany/ct-images
```

下载完成后，将文件夹 `CT/` 放置在 `data/` 文件夹下，保证文件结构为：

```bash
data/
├── CT
│   ├── cancer
│   └── normal
└── readme.md
```