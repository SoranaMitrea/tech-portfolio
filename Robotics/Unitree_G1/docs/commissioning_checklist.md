# Commissioning Checklist: Development PC ↔ Unitree G1

Connecting an Ubuntu 22.04 development PC to the robot's internal network via a USB-Ethernet adapter, setting up CycloneDDS and checking the connection.

## Safety first

- The robot weighs about 35 kg. While working at the PC, it hangs secured in its frame, feet free, wheels locked.
- Emergency stop on the remote control: **L2 + B** for at least 5 seconds.

## 1. Identify the network interface

```bash
ip link show
```

The USB-Ethernet adapter appears with a name starting with `enx`, derived from its MAC address (e.g. `enx607d0912021e`). Replace this name in all following commands.

## 2. Set address and multicast route permanently

```bash
nmcli con mod enx607d0912021e ipv4.method manual ipv4.addresses 192.168.123.51/24 ipv4.never-default yes
nmcli con mod enx607d0912021e +ipv4.routes "224.0.0.0/4"
nmcli con up enx607d0912021e
```

- `ipv4.method manual`: no DHCP server runs on the robot network.
- `ipv4.never-default yes`: prevents the adapter from taking over the default route and breaking the Wi-Fi internet connection.
- `224.0.0.0/4`: tells the kernel to send multicast over this interface – DDS needs it.

## 3. Disable the reverse path filter

With several interfaces (Wi-Fi, Docker bridge, USB-Ethernet), the kernel may drop incoming multicast packets.

```bash
sudo sysctl -w net.ipv4.conf.all.rp_filter=0
sudo sysctl -w net.ipv4.conf.enx607d0912021e.rp_filter=0
echo "net.ipv4.conf.all.rp_filter=0" | sudo tee /etc/sysctl.d/99-g1-dds.conf
```

After `nmcli con up`, the per-interface value is reset – set it again in the running session.

## 4. Verify

| Check | Command | Expected |
|---|---|---|
| Address | `ip addr show enx607d0912021e` | `inet 192.168.123.51/24` |
| Multicast route | `ip route \| grep 224` | `224.0.0.0/4 dev enx607d0912021e` |
| rp_filter | `sysctl net.ipv4.conf.all.rp_filter` | `= 0` |
| Reachability | `ping -c 3 192.168.123.161` | 0% packet loss |

## 5. Configure CycloneDDS

Save [`cyclonedds_g1.xml`](../examples/cyclonedds_g1.xml) as `~/.config/cyclonedds/g1.xml`, then:

```bash
export CYCLONEDDS_URI=file://$HOME/.config/cyclonedds/g1.xml
```

- The URI needs three slashes in total (`file:///home/...`). With two, CycloneDDS looks for a host named "home".
- The variable only applies to the current terminal.

## 6. Test the connection

```bash
~/unitree_sdk2/build/bin/g1_termination
```

`g1_termination` is a safety watchdog, not a connection test: it only prints on errors.

| Observation | Meaning |
|---|---|
| One "Lost connection!" at start, then silence | Normal – DDS discovery takes one to three seconds |
| Repeated "Lost connection!" every second | Real connection problem |

**Silence means success.**

## 7. Read sensor data

Build and run [`read_state.cpp`](../examples/read_state.cpp). It only subscribes and sends nothing.

## Lessons learned

| Problem | Cause | Fix |
|---|---|---|
| No data after reboot or replugging | IPv4 address on the USB adapter lost | Check `ip addr` at the start of every session, `nmcli con up` if needed |
| Repeated "Lost connection!" although the network is fine | Two DDS programs running at the same time | `ps aux \| grep -E 'g1_\|test_'`, stop the extra program |
| "can't open configuration file" | `file://` with only two slashes | Use `file:///` |
| Wi-Fi internet stops working | Adapter took over the default route | `ipv4.never-default yes` |
| PC cannot find the robot over Wi-Fi | DDS multicast does not cross a router | Use Ethernet in the same network segment |
