from datetime import datetime


def to_datetime(time: str) -> datetime:
    return datetime.strptime(time, '%Y-%m-%d %H:%M:%S')


def get_day_seconds() -> float:
    return 60.0 * 60.0 * 24.0


def get_year_seconds() -> float:
    return 365.2425 * get_day_seconds()
