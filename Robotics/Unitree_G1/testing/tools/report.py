#!/usr/bin/env python3
"""Builds report/test_report.md from the JUnit file of pytest and requirements.md.

    python3 tools/report.py report/junit.xml requirements.md report/test_report.md

Traceability: requirement -> tests -> result. Expected failures (xfail) are listed as findings.
"""
import datetime, re, sys
import xml.etree.ElementTree as ET


def anforderungen(pfad):
    reqs = {}
    for zeile in open(pfad, encoding='utf-8'):
        m = re.match(r'\|\s*(REQ-\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|', zeile)
        if m:
            reqs[m.group(1)] = {'text': m.group(2).strip(), 'methode': m.group(4).strip()}
    return reqs


def ergebnisse(pfad):
    tests = []
    for tc in ET.parse(pfad).getroot().iter('testcase'):
        req = [p.get('value') for p in tc.iter('property') if p.get('name') == 'req']
        if tc.find('failure') is not None or tc.find('error') is not None:
            status, info = 'failed', ''
        elif tc.find('skipped') is not None:
            s = tc.find('skipped')
            status = 'finding' if s.get('type') == 'pytest.xfail' else 'skipped'
            info = (s.get('message') or '').replace('xfail-marked test passes unexpectedly', '')
        else:
            status, info = 'passed', ''
        tests.append({'name': tc.get('name'), 'req': req[0] if req else '-', 'status': status, 'info': info})
    return tests


def main(junit, req_datei, ausgabe):
    reqs, tests = anforderungen(req_datei), ergebnisse(junit)
    z = ['# Test report – G1 goal navigation', '',
         'Generated %s UTC from `%s`.' % (datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M'), junit), '',
         '| Status | Count |', '|---|---|']
    for s in ('passed', 'failed', 'finding', 'skipped'):
        z.append('| %s | %d |' % (s, sum(t['status'] == s for t in tests)))
    z += ['', '## Traceability', '', '| ID | Requirement | Method | Tests | Passed | Findings | Result |',
          '|---|---|---|---|---|---|---|']
    for rid, r in reqs.items():
        eigene = [t for t in tests if t['req'] == rid]
        ok = sum(t['status'] == 'passed' for t in eigene)
        fund = sum(t['status'] == 'finding' for t in eigene)
        fehl = sum(t['status'] == 'failed' for t in eigene)
        if not eigene:
            ergebnis = 'see method'
        elif fehl:
            ergebnis = 'FAILED'
        elif fund:
            ergebnis = 'passed, with open finding'
        else:
            ergebnis = 'passed'
        z.append('| %s | %s | %s | %d | %d | %d | %s |' % (rid, r['text'], r['methode'], len(eigene), ok, fund, ergebnis))
    funde = [t for t in tests if t['status'] == 'finding']
    if funde:
        z += ['', '## Findings (open, documented as expected failures)', '']
        for t in funde:
            z.append('- **%s** `%s`: %s' % (t['req'], t['name'], t['info'].strip()))
    fehler = [t for t in tests if t['status'] == 'failed']
    if fehler:
        z += ['', '## Failed tests', ''] + ['- **%s** `%s`' % (t['req'], t['name']) for t in fehler]
    open(ausgabe, 'w', encoding='utf-8').write('\n'.join(z) + '\n')
    print('\n'.join(z))


if __name__ == '__main__':
    main(*sys.argv[1:4])
