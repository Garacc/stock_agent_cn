# ChangeLog

## [v1.1.0] - 2025-03-03
### Brief
stock_agent_cn v1.1.0版本，当前代码已经可以在常规上网环境下运行，且全部agent均已通过测试。
**CallOut**
- 第一个完全跑通全部agent的版本，已可用于实盘辅助决策；
- 当前的核心功能为，基于给定的单个股票\[ticker\]，结合近期的技术指标和市场情绪，给出相应的交易信号。

### Add
**Function**
- UnitTest：完善了各类子函数的单测流程，确保功能正常；
- Logger：独立Logger，并将几乎全部打印信息写入Logger中，方便后续信息回溯与判断；
- MultiModel Compatibility：添加了多模型的兼容功能，支持同时使用多个模型进行决策；
**Docs**
- ChangeLog：添加了ChangeLog，方便后续版本更新的记录。

### Changes
**Function**
- CommonChanges：优化了部分代码，提高了代码的可读性和可维护性。

## [v1.0.0] - 2025-03-02
### Brief
stock_agent_cn初始版本，基于ai-hedge-fund和24mlight的改造版进行开发，支持基本的A股功能。

### Changes
**Function**
- ReOrg Code：对代码的结构进行了初步的调整，修改部分readme以保证项目可以正确配置和运行。