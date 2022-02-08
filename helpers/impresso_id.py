import datetime


def is_page(impresso_id):
    return impresso_id[-5] == 'p'


def get_page(impresso_id):
    if not is_page(impresso_id):
        raise ValueError(f"{impresso_id} does not refer to a page")

    return int(impresso_id[-4:])


def is_front_page(impresso_id):
    return get_page(impresso_id) == 1


def get_day(impresso_id):
    return int(impresso_id[-10:-8])


def get_month(impresso_id):
    return int(impresso_id[-13:-11])


def get_year(impresso_id):
    return int(impresso_id[-18:-14])


def get_journal(impresso_id):
    return impresso_id[:-19]


def get_edition(impresso_id):
    return impresso_id[-7]


def get_meta_issue_id(impresso_id):
    return impresso_id[:-6]


def get_date(impresso_id):
    return datetime.datetime(get_year(impresso_id), get_month(impresso_id), get_day(impresso_id))
