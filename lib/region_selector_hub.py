import settings


def select_regions_to_evaluate(regions_used):

    list_regions = []
    if regions_used == "all":
        list_regions = range(1,117,1)
    elif regions_used == "most important":
        list_regions = settings.list_regions_evaluated
    elif regions_used == "three":
        list_regions = [1,2,3]

    return list_regions