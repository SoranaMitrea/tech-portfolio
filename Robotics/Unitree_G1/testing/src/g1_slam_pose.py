#!/usr/bin/env python3
"""g1_slam_pose.py - liest die Pose des Roboters in der Karte und schreibt sie nach /tmp/g1_pose.json.

Rein lesend: Das Programm abonniert nur das Topic /slam_info (std_msgs/String, JSON) und sendet nichts.
Es nimmt die Meldungen vom Typ pos_info (etwa 1,2 Hz) und schreibt Position und Blickrichtung.

Starten (ein Terminal, ROS 2 Humble wie bei 'ros2 topic echo'):
    source /opt/ros/humble/setup.bash
    python3 g1_slam_pose.py

Ausgabedatei /tmp/g1_pose.json, Beispiel:
    {"x": -0.39, "y": 0.33, "z": 0.06, "yaw_grad": 177.3, "q_w": ..., "q_x": ..., "q_y": ..., "q_z": ...,
     "karte": "dGVzdDAy_20261007", "zeit": 1791379014.5}
x, y, z und yaw_grad gelten im Koordinatensystem der Roboterkarte. zeit ist die Uhr dieses Rechners.

Es darf nur eine Bruecke laufen. Eine zweite bricht beim Start mit einer Meldung ab (Sperrdatei
/tmp/g1_slam_pose.lock). Mehrere gleichzeitige Empfaenger haben am 14.10.2026 die Pose blockiert.
Kommt laenger als 5 s keine pos_info, meldet das Programm das im Terminal.
"""
import fcntl, json, math, os, sys, time

AUSGABE = sys.argv[1] if len(sys.argv) > 1 else '/tmp/g1_pose.json'

def yaw_aus_quaternion(w, x, y, z):
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

def pose_aus_meldung(text):
    """Gibt ein Dict mit der Pose zurueck, oder None, wenn die Meldung keine pos_info ist."""
    try:
        d = json.loads(text)
    except ValueError:
        return None
    if d.get('type') != 'pos_info':
        return None
    try:
        p = d['data']['currentPose']
        w, qx, qy, qz = float(p['q_w']), float(p['q_x']), float(p['q_y']), float(p['q_z'])
        return {'x': float(p['x']), 'y': float(p['y']), 'z': float(p['z']),
                'yaw_grad': yaw_aus_quaternion(w, qx, qy, qz),
                'q_w': w, 'q_x': qx, 'q_y': qy, 'q_z': qz,
                'karte': d['data'].get('pcdName', '')}
    except (KeyError, TypeError, ValueError):
        return None

def schreiben(pose):
    pose = dict(pose)
    pose['zeit'] = time.time()
    tmp = AUSGABE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(pose, f)
    os.replace(tmp, AUSGABE)      # atomar, damit andere Programme nie eine halbe Datei lesen

SPERRE = '/tmp/g1_slam_pose.lock'

def sperren():
    """Nur eine Bruecke zur Zeit. Gibt die offene Sperrdatei zurueck oder beendet das Programm."""
    f = open(SPERRE, 'a+')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.seek(0)
        pid = f.read().strip() or 'unbekannt'
        sys.stderr.write('ABBRUCH: Es laeuft schon eine Pose-Bruecke (PID %s).\n'
                         'Pruefen: pgrep -af g1_slam_pose   Beenden: pkill -f g1_slam_pose.py\n' % pid)
        sys.exit(1)
    f.seek(0); f.truncate(); f.write(str(os.getpid())); f.flush()
    return f

def main():
    sperre = sperren()
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class Knoten(Node):
        def __init__(self):
            super().__init__('g1_slam_pose')
            self.n = 0
            self.letzte = 0.0
            self.zuletzt_empfangen = time.time()
            self.gewarnt = False
            self.create_subscription(String, '/slam_info', self.cb, 10)
            self.create_timer(5.0, self.waechter)
            self.get_logger().info('Warte auf pos_info von /slam_info, schreibe nach ' + AUSGABE)
        def waechter(self):
            ruhe = time.time() - self.zuletzt_empfangen
            if ruhe > 5.0:
                self.gewarnt = True
                self.get_logger().warn('Seit %.0f s keine pos_info (Meldungen bisher: %d). Lokalisiert? Nur eine Bruecke? pgrep -af g1_slam_pose' % (ruhe, self.n))
        def cb(self, msg):
            p = pose_aus_meldung(msg.data)
            if p is None:
                return
            schreiben(p)
            self.n += 1
            self.zuletzt_empfangen = time.time()
            if self.gewarnt:
                self.gewarnt = False
                self.letzte = 0.0
            jetzt = time.time()
            if jetzt - self.letzte > 5.0:
                self.letzte = jetzt
                self.get_logger().info('Pose x %.2f y %.2f Blick %.0f Grad (Meldungen: %d)' % (p['x'], p['y'], p['yaw_grad'], self.n))

    rclpy.init()
    k = Knoten()
    try:
        rclpy.spin(k)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            k.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        sperre.close()

if __name__ == '__main__':
    main()
