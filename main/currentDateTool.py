import datetime
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple


class CurrentDateTool:
    """本地当前日期工具类"""

    @classmethod
    def get_current_date(cls, format: str = None) -> Dict[str, Any]:
        """
        获取系统当前日期信息

        Args:
            format: 日期格式，可选参数
                   支持格式：YYYY-MM-DD, YYYY-MM-DD HH:mm:ss,
                           YYYY/MM/DD, DD-MM-YYYY, 等

        Returns:
            日期信息字典
        """
        try:
            now = datetime.now()
            current_date = date.today()  # 修正这里：datetime.date.today() -> date.today()

            # 如果没有指定格式，返回多种格式
            if not format:
                result = {
                    "iso_format": now.isoformat(),
                    "date_only": current_date.isoformat(),
                    "readable_format": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "chinese_format": now.strftime("%Y年%m月%d日 %H时%M分%S秒"),
                    "timestamp": int(now.timestamp()),
                    "weekday": now.strftime("%A"),
                    "weekday_chinese": cls._get_chinese_weekday(now.weekday()),
                    "day_of_year": now.timetuple().tm_yday,
                    "is_weekend": now.weekday() >= 5
                }
                return {
                    "success": True,
                    "result": json.dumps(result, ensure_ascii=False, indent=2),
                    "current_time": now.isoformat()
                }

            # 处理指定格式
            format = format.upper().strip()

            # 预定义格式映射
            format_mapping = {
                "YYYY-MM-DD": "%Y-%m-%d",
                "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
                "YYYY-MM-DD HH:MM": "%Y-%m-%d %H:%M",
                "YYYY/MM/DD": "%Y/%m/%d",
                "DD-MM-YYYY": "%d-%m-%Y",
                "MM/DD/YYYY": "%m/%d/%Y",
                "YYYY年MM月DD日": "%Y年%m月%d日",
                "YYYY年MM月DD日 HH时MM分SS秒": "%Y年%m月%d日 %H时%M分%S秒",
                "FULL": "%A, %B %d, %Y %H:%M:%S",
                "RFC_822": "%a, %d %b %Y %H:%M:%S",
                "ISO": "%Y-%m-%dT%H:%M:%S"
            }

            # 获取对应的格式字符串
            format_str = format_mapping.get(format, format)

            try:
                formatted_date = now.strftime(format_str)
                result = {
                    "formatted_date": formatted_date,
                    "format_used": format_str,
                    "original_format": format,
                    "timestamp": int(now.timestamp()),
                    "iso_format": now.isoformat()
                }

                return {
                    "success": True,
                    "result": json.dumps(result, ensure_ascii=False),
                    "current_time": now.isoformat()
                }

            except ValueError as e:
                return {
                    "success": False,
                    "result": f"日期格式错误: {str(e)}，支持的格式: {', '.join(format_mapping.keys())}"
                }

        except Exception as e:
            return {
                "success": False,
                "result": f"获取日期信息失败: {str(e)}"
            }

    @staticmethod
    def _get_chinese_weekday(weekday: int) -> str:
        """获取中文星期几"""
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return weekdays[weekday]