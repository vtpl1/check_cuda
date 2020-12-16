import collections
from typing import Any, Dict, List, Union


def get_dict_or_list(in_dict1: Any) -> Any:
    in_dict = in_dict1
    try:
        in_dict = vars(in_dict1)
    except TypeError:
        pass
    return in_dict


def get_flatten_dict(in_dict1: Union[Dict[str, Any], List[Any]], in_key: str = None) -> Dict[str, str]:
    ret = dict()

    in_dict_or_list = get_dict_or_list(in_dict1)
    if isinstance(in_dict_or_list, collections.Mapping):
        for k, v in in_dict_or_list.items():
            v1 = get_dict_or_list(v)
            if isinstance(v1, collections.Mapping) or isinstance(v1, list):
                is_empty_dict = True  # special care for empty dictionary as values
                for k_i, v_i in get_flatten_dict(v1, k).items():
                    key_i = k_i if in_key is None else in_key + '.' + k_i
                    ret[key_i] = v_i
                    is_empty_dict = False
                if is_empty_dict:
                    key = k if in_key is None else in_key + '.' + k
                    ret[key] = k
            else:
                key = k if in_key is None else in_key + '.' + k
                ret[key] = k
    elif isinstance(in_dict_or_list, list):
        i = 0
        for v in in_dict_or_list:
            i += 1
            k = 'i#' + str(i)
            v1 = get_dict_or_list(v)
            if isinstance(v1, collections.Mapping) or isinstance(v1, list):
                is_empty_dict = True  # special care for empty dictionary as values
                for k_i, v_i in get_flatten_dict(v1, k).items():
                    key_i = k_i if in_key is None else in_key + '.' + k_i
                    ret[key_i] = 'i#' + str(i) + v_i
                    is_empty_dict = False
                if is_empty_dict:
                    key = k if in_key is None else in_key + '.' + k
                    ret[key] = k
            # else:
            #     key = k if in_key is None else in_key + '.' + k
            #     ret[key] = k
    return ret


def get_flatten_keys(obj=None) -> Dict[str, str]:
    ret = {}
    if obj is not None:
        d = obj
        try:
            d = vars(obj)
        except TypeError:
            pass
        ret = get_flatten_dict(d)
    return ret


def get_flatten_keys_list(obj=None, ignore_list=[]) -> List[str]:
    ret = []
    for k, v in get_flatten_keys(obj).items():
        if k not in ignore_list:
            ret.append(v)
    # print(ret)
    return ret
