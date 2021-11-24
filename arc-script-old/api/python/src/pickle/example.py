from __future__ import unicode_literals

import os
import shutil
import sys
import tempfile
import unittest

import pytest

import cloudpickle
from cloudpickle.compat import pickle


class CloudPickleFileTests(unittest.TestCase):
    """In Cloudpickle, expected behaviour when pickling an opened file
    is to send its contents over the wire and seek to the same position."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmpfilepath = os.path.join(self.tmpdir, 'testfile')
        self.teststring = 'Hello world!'

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_empty_file(self):
        # Empty file
        open(self.tmpfilepath, 'w').close()
        with open(self.tmpfilepath, 'r') as f:
            self.assertEqual('', pickle.loads(cloudpickle.dumps(f)).read())
        os.remove(self.tmpfilepath)
