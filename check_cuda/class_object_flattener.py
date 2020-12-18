import collections
from pprint import pprint
from typing import Any, Dict, List, Union


def get_dict_or_list(in_dict1: Any) -> Any:
    in_dict = in_dict1
    try:
        in_dict = vars(in_dict1)
    except TypeError:
        pass
    return in_dict


def get_str(in_key: Any) -> Union[str, None]:
    return str(in_key) if in_key is not None else None

def get_flatten_dict(in_dict1: Union[Dict[str, Any], List[Any]], in_key: str = None) -> Dict[str, str]:
    ret = dict()
    in_key = get_str(in_key)
    in_dict_or_list = get_dict_or_list(in_dict1)
    if isinstance(in_dict_or_list, collections.Mapping):
        for k, v in in_dict_or_list.items():
            v1 = get_dict_or_list(v)
            if isinstance(v1, collections.Mapping) or isinstance(v1, list):
                # is_empty_dict = True  # special care for empty dictionary as values
                for k_i, v_i in get_flatten_dict(v1, k).items():
                    key_i = k_i if in_key is None else in_key + '.' + k_i
                    ret[key_i] = v_i
                    is_empty_dict = False
                # if is_empty_dict:
                #     k = get_str(k)
                #     key = k if in_key is None else in_key + '.' + k
                #     ret[key] = k
            else:
                k = get_str(k)
                key = k if in_key is None else in_key + '.' + k
                ret[key] = k
    elif isinstance(in_dict_or_list, list):
        i = 0
        for v in in_dict_or_list:            
            k = '#' + str(i)
            v1 = get_dict_or_list(v)
            if isinstance(v1, collections.Mapping) or isinstance(v1, list):
                is_empty_dict = True  # special care for empty dictionary as values
                for k_i, v_i in get_flatten_dict(v1, k).items():
                    key_i = k_i if in_key is None else in_key + '.' + k_i
                    ret[key_i] = '#' + str(i) + v_i
                    is_empty_dict = False
                if is_empty_dict:
                    key = k if in_key is None else in_key + '.' + k
                    ret[key] = k
            # else:
            #     key = k if in_key is None else in_key + '.' + k
            #     ret[key] = k
            i += 1
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


def get_flatten_keys_list(in_dict: Dict[str, str], ignore_list=[]) -> List[str]:
    ret = []
    for k, v in in_dict.items():
        if k not in ignore_list:
            ret.append(v)
    return ret

def get_value_for_key(obj: Any, in_key: str) -> Any:
    keys = in_key.split('.', maxsplit=1)
    k = keys[0]
    if k.startswith('#'):
        indx = int(k[1:])
        obj1 = obj[indx]
    else:        
        try:
            k = int(k)  #to take care of dictionary having integer as key
        except ValueError:
            pass
        
        obj1 = get_dict_or_list(obj).get(k)

    if len(keys) > 1:
        return get_value_for_key(obj1, keys[1])
    else:
        return obj1


def get_flatten_values_list(obj: Any, in_dict: Dict[str, str], ignore_list=[]) -> List[Any]:
    ret = []
    for k, v in in_dict.items():
        if k not in ignore_list:
            ret.append(get_value_for_key(obj, k))
    return ret
