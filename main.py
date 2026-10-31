from astrbot.api.star import Context, Star, register


@register(
    "astrbot_plugin_tool_context",
    "Moebius1453",
    "工具调用日志归档与上下文优化（开发中）",
    "0.1.0",
)
class ToolContextPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
