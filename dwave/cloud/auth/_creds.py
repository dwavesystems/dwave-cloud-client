# Copyright 2023 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# `diskcache.Cache` subclass that allows per-instance db name override.

# TODO: submit a PR for this change upstream, and then remove it from here.

import os
import os.path as op
import sqlite3

import diskcache


class _Cache(diskcache.Cache):

    def __init__(self, dbname=diskcache.core.DBNAME, **kwargs):
        self.dbname = dbname
        super().__init__(**kwargs)

    @property
    def _con(self):
        # copied from `diskcache.core.Cache._con` and modified to enable
        # db name per-instance override

        # Check process ID to support process forking. If the process
        # ID changes, close the connection and update the process ID.

        local_pid = getattr(self._local, 'pid', None)
        pid = os.getpid()

        if local_pid != pid:
            self.close()
            self._local.pid = pid

        con = getattr(self._local, 'con', None)

        if con is None:
            con = self._local.con = sqlite3.connect(
                op.join(self._directory, self.dbname),
                timeout=self._timeout,
                isolation_level=None,
            )

            # Some SQLite pragmas work on a per-connection basis so
            # query the Settings table and reset the pragmas. The
            # Settings table may not exist so catch and ignore the
            # OperationalError that may occur.

            try:
                select = 'SELECT key, value FROM Settings'
                settings = con.execute(select).fetchall()
            except sqlite3.OperationalError:
                pass
            else:
                for key, value in settings:
                    if key.startswith('sqlite_'):
                        self.reset(key, value, update=False)

        return con
