# 源类别与问题模板约定

Active 表示为探测或通信主动发射的信号，Passive 表示舰船或水下航行目标自身辐射的噪声。分类依据是声源产生机制，不是接收机工作模式。

共享实现位于 `testsite/core/source_label_prompts.py` 和 `testsite/core/shared_terminology.py`；顶层同名文件为兼容导入入口。两个 Step 2 使用共享实现，并将问题、参考答案、标准标签和元数据写入 JSONL。

当前提示词版本为 `source_origin_v4_best_match_letter_only`。保留多套问题措辞，末尾统一要求选择最合理选项并只输出大写字母。评测接收端兼容单个 ASCII 小写字母，参考答案仍为大写。第三轮提示词和参考解释不因这次接收规则调整而变化。

正式评测使用记录中的问题，不从配置随机重建。更新代码不会自动修改既有 JSONL。需要新问题时，可在 `archive/` 下运行两个 Step 2，配置新的 `qa_output` 目录保存结果，随后更新筛选和导出产物，无须仅为问题措辞重新运行声学仿真。

详见 [数据发布说明](testsite/DATASET_EVALUATION.md) 和 [评估框架](testsite/评估框架设计文档.md)。
