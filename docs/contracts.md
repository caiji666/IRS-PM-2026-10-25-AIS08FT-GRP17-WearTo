# 接口契约 v0.1

`TripRequest` 保存用户提供的事实。缺失字段保留空值，由控制层追问，不编造预算、日期或天气。

`FashionContext` 分离硬约束、软偏好与证据。硬约束属性未知时，不能自动当作满足。

`ProductCandidate` 区分 owned/new，金额为最小货币单位整数；owned 新购成本为 0。库存和属性未知应保持 unknown。所有参与金额计算的条目必须使用相同币种，货币转换需单独实现。

`OutfitResult` 是推荐数据传输结构；`constraint_status` 默认 unverified。schema 通过仅说明格式合法，不证明预算、库存、尺码或天气满足。

`ModuleResult` 统一返回 ok / needs_input / no_candidates / not_implemented / error；附带 data 与 reason。目前工具全部是 not_implemented 骨架，不能将其当作成功。

预算修改只更新明确提供的字段；“便宜一点”属于排序偏好或需要追问，不能生成新预算。Top-K 无解仅表示当前候选池无解。

各模块 PR 应提供序列化样例，注明 currency、单位、字段含义、来源与版本。新增必填字段前与上下游协商。
