import os, subprocess, sys
import pytest

HIER = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HIER), 'src')
sys.path.insert(0, SRC)


def pytest_configure(config):
    config.addinivalue_line('markers', 'req(id): requirement ID covered by this test (see requirements.md)')


def pytest_collection_modifyitems(items):
    # Write the requirement ID into the JUnit report, so tools/report.py can build the traceability table
    for item in items:
        for m in item.iter_markers('req'):
            item.user_properties.append(('req', m.args[0]))


@pytest.fixture(scope='session')
def ziel_cli(tmp_path_factory):
    """Compiles the C++ goal checks once per test run and returns a function that calls them."""
    exe = str(tmp_path_factory.mktemp('bin') / 'ziel_cli')
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-o', exe, os.path.join(SRC, 'ziel_cli.cpp')], check=True)

    def aufrufen(zx, zy, pose_json, jetzt, fsm, max_m=3.0, *optionen):
        r = subprocess.run([exe, str(zx), str(zy), pose_json, str(jetzt), str(fsm), str(max_m), *optionen],
                           capture_output=True, text=True)
        return r.returncode, r.stdout.strip()
    return aufrufen
