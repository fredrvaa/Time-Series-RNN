from datetime import datetime


def to_datetime(time: str) -> datetime:
    return datetime.strptime(time, '%Y-%m-%d %H:%M:%S')