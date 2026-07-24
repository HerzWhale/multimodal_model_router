# 文本主分类人工评估报告

说明：本报告只评估文本文件的 `topic` 主分类，不评估摘要、关键词、副分类、图片或视频结果。

| 指标 | 数值 | 含义 |
|---|---:|---|
| 模板行数 | 18 | 需要人工判断的文本样本数 |
| 已评估样本数 | 18 | 已填写 gold_topic 的样本数 |
| 缺少标签样本数 | 0 | 尚未填写 gold_topic 的样本数 |
| 有效预测数 | 17 | 成功产出九类范围内 predicted_topic 的样本数 |
| 缺少预测数 | 1 | 调用或解析失败导致没有 predicted_topic 的样本数 |
| 非法预测数 | 0 | predicted_topic 不属于九类允许值的样本数 |
| 预测正确数 | 17 | predicted_topic 与 gold_topic 相同的样本数 |
| 端到端 Accuracy | 94.44% | 以全部已标注样本为分母，无有效预测按未命中计算 |
| 有效预测 Accuracy | 100.00% | 仅衡量成功产出九类预测的样本 |
| 预测覆盖率 | 94.44% | 有效预测数占已标注样本数的比例 |
| Macro-F1 | 96.30% | 九类业务标签中本批次实际出现分类 F1 的简单平均 |

## 分类级指标

| topic | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| ads_marketing | 2 | 100.00% | 100.00% | 100.00% |
| entertainment | 2 | 100.00% | 100.00% | 100.00% |
| finance_business | 2 | 100.00% | 100.00% | 100.00% |
| knowledge | 2 | 100.00% | 100.00% | 100.00% |
| lifestyle | 2 | 100.00% | 100.00% | 100.00% |
| news | 2 | 100.00% | 100.00% | 100.00% |
| other | 2 | 100.00% | 100.00% | 100.00% |
| sports_health | 2 | 100.00% | 50.00% | 66.67% |
| technology | 2 | 100.00% | 100.00% | 100.00% |

## 明细

| file_id | file_name | predicted_topic | gold_topic | 是否正确 | 备注 |
|---|---|---|---|---|---|
| file_0001 | 01_news_city_transport.txt | news | news | 是 | 核心是暴雨后地铁恢复运营和限流通报 属于公共事件新闻资讯 |
| file_0002 | 02_finance_ai_chip_market.txt | finance_business | finance_business | 是 | 虽然出现 AI 芯片和云厂商 但重点是财报 资本开支 订单和投资风险 属于财经商业 |
| file_0003 | 03_ads_phone_campaign.txt | ads_marketing | ads_marketing | 是 | 内容明确说明品牌合作并引导优惠转化 属于广告营销 |
| file_0004 | 04_technology_ai_agent_update.txt | technology | technology | 是 | 核心是 AI Agent 软件功能 工作流和工具调用日志 属于科技数码 |
| file_0005 | 05_sports_health_home_training.txt | sports_health | sports_health | 是 | 核心是久坐人群训练动作 运动风险和健康建议 属于体育健康 |
| file_0006 | 06_entertainment_movie_variety.txt | entertainment | entertainment | 是 | 核心是综艺节目 嘉宾互动 舞台花絮和粉丝讨论 属于娱乐休闲 |
| file_0007 | 07_lifestyle_travel_food.txt | lifestyle | lifestyle | 是 | 核心是个人旅行 美食和周末生活体验 属于生活日常 |
| file_0008 | 08_knowledge_history_science.txt | knowledge | knowledge | 是 | 核心是解释古城与河流关系的泛知识内容 没有更强领域归属 属于知识科普 |
| file_0009 | 09_finance_tech_ai_cloud_capex.txt | finance_business | finance_business | 是 | 虽然出现 AI 云服务 芯片和推理成本 但核心是云厂商收入结构 竞争策略和利润压力 属于财经商业 |
| file_0010 | 10_ads_disguised_phone_review.txt | ads_marketing | ads_marketing | 是 | 形式像手机测评 但明确品牌合作 购买入口和促销转化 应归入广告营销 |
| file_0011 | 11_news_entertainment_film_festival_incident.txt | news | news | 是 | 虽然出现电影节 明星和粉丝讨论 但核心是活动临时取消和官方回应 属于新闻资讯 |
| file_0012 | 12_knowledge_with_soft_ad_nutrition.txt | knowledge | knowledge | 是 | 虽然出现产品露出 但主体是解释配料表和控糖知识 属于知识科普 |
| file_0013 | 13_lifestyle_commerce_travel_hotel.txt | lifestyle | lifestyle | 是 | 虽然出现房型价格和链接 但主体是个人旅行休息体验 属于生活日常 |
| file_0014 | 14_sports_health_public_event_marathon.txt |  | sports_health | 否 | 虽然开头像赛事事件 但主体是跑者补给和运动健康建议 属于体育健康 |
| file_0015 | 15_technology_long_local_ai_workflow.txt | technology | technology | 是 | 核心是本地 AI 工作流的数据格式 任务队列 结构校验和调用安全 属于科技工程内容 |
| file_0016 | 16_entertainment_long_variety_production.txt | entertainment | entertainment | 是 | 核心是综艺剪辑 嘉宾互动 节目效果和观众讨论 属于娱乐休闲 |
| file_0017 | 17_other_long_campus_lost_found.txt | other | other | 是 | 核心是校内失物认领事务和操作流程 不符合其余八类业务主题 |
| file_0018 | 18_other_long_community_coordination.txt | other | other | 是 | 核心是社区共享工具借用与安全登记流程 不符合其余八类业务主题 |

## 没有有效预测的文件

- file_0014

## 字段说明

| 字段 | 含义与作用 |
|---|---|
| `gold_topic` | 人工标注的正确主分类，用于和模型预测的 predicted_topic 对比。 |
| `predicted_topic` | 模型输出的主分类，用于衡量文本分类结果是否命中人工标签。 |
| `reviewer_note` | 人工评审备注，用于记录分类正确或错误的判断依据。 |
| `accuracy` | 端到端文本主分类准确率，计算方式为 correct_count / evaluated_count；调用失败或无有效预测按未命中计算。 |
| `valid_prediction_accuracy` | 仅在有效九分类预测中的准确率，用于把分类判断能力与调用可用性分开观察。 |
| `prediction_coverage` | 有效九分类预测数占已标注样本数的比例，用于衡量模型调用和结构解析是否稳定。 |
| `macro_f1` | 九类业务标签中本批次实际出现分类的 F1 简单平均；无结果和非法预测会造成对应真实分类漏报，但不会被当作新分类。 |
| `precision` | 预测为某分类的样本中，真正属于该分类的比例。 |
| `recall` | 人工标注为某分类的样本中，被模型正确识别的比例。 |
| `f1` | 单个分类 Precision 与 Recall 的调和平均，用于综合衡量误报和漏报。 |
| `support` | 人工标准答案中属于某分类的样本数，用于判断该分类证据量。 |
| `evaluated_labels` | 人工标签或模型预测中实际出现的分类，用于说明本次指标覆盖范围。 |
| `evaluated_count` | 已经填写 gold_topic 并纳入统计的文本样本数。 |
| `valid_prediction_count` | 成功产出九类范围内 predicted_topic 的样本数。 |
| `missing_prediction_count` | 没有产出 predicted_topic 的已标注样本数，常用于识别调用或解析失败。 |
| `invalid_prediction_count` | 产出了内容但不属于九类允许值的样本数，用于发现输出约束失效。 |
