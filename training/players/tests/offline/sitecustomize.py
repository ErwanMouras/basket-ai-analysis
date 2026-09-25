"""Validation subprocess guard, enabled only by its explicit PYTHONPATH.

Block Python Internet sockets, including libraries that swallow download errors.
This is a test tripwire, not an OS network sandbox. Unix sockets remain available.
"""
import socket


def denied(*args, **kwargs):
    raise OSError("Network disabled during synthetic players validation")


_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        return denied()
    return _connect(self, address)


def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        return denied()
    return _connect_ex(self, address)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.create_connection = denied
socket.getaddrinfo = denied
