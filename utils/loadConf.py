#-*- coding: utf-8 -*-
# 
# __author__ : jianxing.wei

import yaml
import io
import json
from util import exceptions
import os

from utils import logger

try:
    # PyYAML version >= 5.1
    # ref: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
    yaml.warnings({'YAMLLoadWarning': False})
except AttributeError:
    pass
#global
conf = {}


def _load_yaml_file(yaml_file):
    """ load yaml file and check file content format
    """
    with io.open(yaml_file, 'r', encoding='utf-8') as stream:
        yaml_content = None
        try:
            yaml_content = yaml.load(stream)
        except yaml.YAMLError as ex:
            logger.log_error(str(ex))
            #raise

        return yaml_content


def _load_json_file(json_file):
    """ load json file and check file content format
    """
    with io.open(json_file, encoding='utf-8') as data_file:
        try:
            json_content = json.load(data_file)
        except json.JSONDecodeError:
            err_msg = u"JSONDecodeError: JSON file format error: {}".format(json_file)
            logger.log_error(err_msg)
            #raise exceptions.FileFormatError(err_msg)

        return json_content

def load_file(file_path):
    if not os.path.isfile(file_path):
        raise exceptions.FileNotFound("{} does not exist.".format(file_path))

    file_suffix = os.path.splitext(file_path)[1].lower()
    if file_suffix == '.json':
        return _load_json_file(file_path)
    elif file_suffix in ['.yaml', '.yml']:
        return _load_yaml_file(file_path)
    else:
        # '' or other suffix
        err_msg = u"Unsupported file format: {}".format(file_path)
        logger.log_warning(err_msg)

        return []

def testload(filename="../remoteDriverConf.yaml"):
    """docstring for testload"""
    global conf
    conf = load_file(filename)
    logger.log_info(json.dumps(conf,indent=4, sort_keys=True))
    logger.log_info(str(conf["remoteDriverURL"]))
    logger.log_info(str(conf["PC"][0]["browerType"]))
    logger.log_info(str(conf["PC"][0]["version"]))
    logger.log_info(str(conf["PC"][1]["browerType"]))
    logger.log_info(str(conf["PC"][1]["version"]))
    logger.log_info('-----end------')


if __name__ == '__main__':
    testload("/Users/wuage/PycharmProjects/fireEye/remoteDriverConf.yaml")