# 数据与目录

最小运行入口接受统一的类别目录：

    DATASET_ROOT/category/train/good/*
    DATASET_ROOT/category/test/good/*
    DATASET_ROOT/category/test/defect_type/*

正常测试目录必须命名为 good；其他测试子目录视为异常。只读取 PNG/JPG/JPEG/BMP/TIF/TIFF。请使用数据集正式发布的划分，不要把测试样本移到训练目录。

MVTec-AD 通常已符合该布局。BTAD、VisA 和 MPDD 必须复用论文实验的数据划分并映射到相同结构。BTAD 的 ok/ko、VisA 的原始 CSV split 等需要明确转换；不要直接将原始 VisA 图像目录传给这个入口。仓库没有声称自动完成原始数据集下载/重划分。

训练正常图路径首先排序，再用 random.Random(seed).shuffle 选择第一张。改变文件名、排序方式或拆分会改变支持图。每次运行把选定的相对路径和 SHA256 写入 run_provenance.json。随机种子为 42/123/999，每个类别独立拟合，正常图一张，不是异常图。transistor 仅原图，其余类别原图加 30 次随机旋转。

类别数量：MVTec-AD 15、BTAD 3、VisA 12、MPDD 6。可从 results/paired_image_auroc.csv 读取准确类别名。获取数据应使用各数据集发布方的渠道，并遵守原始许可。原始数据、权重均不放入该仓库。
