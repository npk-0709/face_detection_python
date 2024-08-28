"""
    # Copyright © 2022 By Nguyễn Phú Khương
    # ZALO : 0363561629
    # Email : dev.phukhuong0709@hotmail.com
    # Github : npk-0709
"""
import string
import hashlib
import random
import os


def md5(data: str):
    return hashlib.md5(data.encode('utf-8')).hexdigest()


def rand_sha1():
    letters = string.ascii_lowercase
    str_rand = ''.join(random.choice(letters) for _ in range(50))
    return hashlib.sha1(str_rand.encode('utf-8')).hexdigest()


def rand_sha265():
    letters = string.ascii_lowercase
    str_rand = ''.join(random.choice(letters) for _ in range(50))
    return hashlib.sha256(str_rand.encode('utf-8')).hexdigest()


def rand_sha512():
    letters = string.ascii_lowercase
    str_rand = ''.join(random.choice(letters) for _ in range(50))
    return hashlib.sha512(str_rand.encode('utf-8')).hexdigest()



def generate_formatted_string(parts: int, length: int, separator: str):
    return separator.join([''.join(random.choices(string.ascii_uppercase + string.digits, k=length)) for _ in range(parts)])


def random_string(string_length, addstring: str = ''):
    letters = string.ascii_letters + string.digits + addstring
    return ''.join(random.choice(letters) for _ in range(string_length))
