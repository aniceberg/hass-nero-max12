"""Amplifier adapter layer for upstream pyxantech plus OSD Nero MAX 12."""

from __future__ import annotations

import logging
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

from pyxantech import (
    AmpControlBase,
    async_get_amp_controller as pyxantech_async_get_amp_controller,
    get_device_config as pyxantech_get_device_config,
)
import serial

LOG = logging.getLogger(__name__)

AMP_TYPE_OSD_NERO_MAX12 = 'osd_nero_max12'

_NERO_CONFIG: dict[str, Any] = {
    'series': AMP_TYPE_OSD_NERO_MAX12,
    'name': 'OSD Audio Nero MAX 12',
    'supports_bass': True,
    'supports_treble': True,
    'supports_balance': True,
    'num_zones': 6,
    'num_sources': 6,
    'max_amps': 1,
    'max_balance': 20,
    'max_bass': 14,
    'max_treble': 14,
    'max_volume': 38,
    'hardware_volume_steps': 38,
    'rs232': {
        'baudrate': 9600,
        'bytesize': 8,
        'parity': 'N',
        'stopbits': 1,
        'timeout': 1.0,
        'write_timeout': 1.0,
    },
    'sources': {
        1: 'Source 1',
        2: 'Source 2',
        3: 'Source 3',
        4: 'Source 4',
        5: 'Source 5',
        6: 'Source 6',
    },
    'zones': {
        11: 'Zone 1',
        12: 'Zone 2',
        13: 'Zone 3',
        14: 'Zone 4',
        15: 'Zone 5',
        16: 'Zone 6',
    },
}

_STATUS_RE = re.compile(
    r'>(?P<zone>\d{2})'
    r'(?P<pa>\d{2})'
    r'(?P<power>\d{2})'
    r'(?P<mute>\d{2})'
    r'(?P<do_not_disturb>\d{2})'
    r'(?P<volume>\d{2})'
    r'(?P<treble>\d{2})'
    r'(?P<bass>\d{2})'
    r'(?P<balance>\d{2})'
    r'(?P<source>\d{2})'
    r'(?P<keypad>\d{2})'
)


@dataclass
class _NeroZoneStatus:
    """NERO zone status parsed into the shape expected by the integration."""

    zone: int
    pa: bool
    power: bool
    mute: bool
    do_not_disturb: bool
    volume: int
    treble: int
    bass: int
    balance: int
    source: int
    keypad: bool

    @classmethod
    def from_response(cls, response: str) -> _NeroZoneStatus | None:
        """Parse a Nero status response, ignoring echoed commands and prompts."""
        match = _STATUS_RE.search(response)
        if not match:
            LOG.debug('Could not parse Nero status response: %r', response)
            return None

        values = match.groupdict()
        return cls(
            zone=int(values['zone']),
            pa=values['pa'] == '01',
            power=values['power'] == '01',
            mute=values['mute'] == '01',
            do_not_disturb=values['do_not_disturb'] == '01',
            volume=int(values['volume']),
            treble=int(values['treble']),
            bass=int(values['bass']),
            balance=int(values['balance']),
            source=int(values['source']),
            keypad=values['keypad'] == '01',
        )

    @property
    def dict(self) -> dict[str, Any]:
        """Return pyxantech-compatible status data."""
        return {
            'zone': self.zone,
            'power': self.power,
            'mute': self.mute,
            'volume': self.volume,
            'treble': self.treble,
            'bass': self.bass,
            'balance': self.balance,
            'source': self.source,
            'paged': False,
            'linked': False,
            'pa': self.pa,
            'do_not_disturb': self.do_not_disturb,
            'keypad': self.keypad,
        }


def get_device_config(
    amp_type: str,
    key: str,
    *,
    log_missing: bool = True,
) -> Any:
    """Return device configuration for Nero or upstream pyxantech amp types."""
    if amp_type == AMP_TYPE_OSD_NERO_MAX12:
        value = _NERO_CONFIG.get(key)
        if value is None and log_missing:
            LOG.warning('No Nero config for key: %s', key)
        return value

    return pyxantech_get_device_config(amp_type, key, log_missing=log_missing)


async def async_get_amp_controller(
    amp_type: str,
    port_url: str,
    loop: asyncio.AbstractEventLoop,
    serial_config_overrides: dict[str, Any] | None = None,
) -> AmpControlBase | None:
    """Create an amp controller, using the local NERO implementation when needed."""
    if amp_type != AMP_TYPE_OSD_NERO_MAX12:
        return await pyxantech_async_get_amp_controller(
            amp_type,
            port_url,
            loop,
            serial_config_overrides,
        )

    controller = NeroMax12Controller(port_url, serial_config_overrides)
    await controller.async_open()
    return controller


class NeroMax12Controller(AmpControlBase):
    """Async controller for the OSD Nero MAX 12 via its HLK-RM04 TCP bridge."""

    def __init__(
        self,
        port_url: str,
        serial_config_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the controller."""
        self._port_url = port_url
        self._serial_config = dict(_NERO_CONFIG['rs232'])
        if serial_config_overrides:
            self._serial_config.update(serial_config_overrides)
        self._port: serial.SerialBase | None = None
        self._lock = asyncio.Lock()

    async def async_open(self) -> None:
        """Open the serial or socket URL."""
        self._port = await asyncio.to_thread(
            serial.serial_for_url,
            self._port_url,
            **self._serial_config,
        )

    def _require_port(self) -> serial.SerialBase:
        if self._port is None:
            raise serial.SerialException('Nero MAX 12 port is not open')
        return self._port

    async def _send(self, command: str, *, expect_status: bool = False) -> str:
        """Send a command with CRLF and return the raw ASCII response."""
        async with self._lock:
            return await asyncio.to_thread(self._send_sync, command, expect_status)

    def _send_sync(self, command: str, expect_status: bool) -> str:
        """Synchronous transport routine run off the event loop."""
        port = self._require_port()
        request = f'{command}\r\n'.encode('ascii')

        port.reset_input_buffer()
        port.reset_output_buffer()
        LOG.debug('Sending Nero command: %r', request)
        port.write(request)
        port.flush()

        chunks = bytearray()
        while True:
            chunk = port.read(1)
            if not chunk:
                break
            chunks += chunk
            if b'Command Error' in chunks:
                break
            if expect_status and b'>' in chunks and chunks.endswith(b'#'):
                break
            if (
                not expect_status
                and chunks.endswith(b'#')
                and len(chunks) > len(request)
            ):
                break

        response = bytes(chunks).decode('ascii', errors='ignore')
        LOG.debug('Received NERO response: %r', response)
        if 'Command Error' in response:
            raise serial.SerialException(f'Nero command failed: {command}')
        return response

    def _validate_zone(self, zone: int) -> None:
        if zone not in _NERO_CONFIG['zones']:
            raise ValueError(f'Invalid Nero MAX 12 zone: {zone}')

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return int(max(minimum, min(value, maximum)))

    async def zone_status(self, zone: int) -> dict[str, Any] | None:
        """Get current status for a zone."""
        self._validate_zone(zone)
        response = await self._send(f'?{zone}', expect_status=True)
        status = _NeroZoneStatus.from_response(response)
        return status.dict if status else None

    async def set_power(self, zone: int, power: bool) -> None:
        """Set zone power state."""
        self._validate_zone(zone)
        await self._send(f'<{zone}PR{int(power):02}')

    async def set_mute(self, zone: int, mute: bool) -> None:
        """Set zone mute state."""
        self._validate_zone(zone)
        await self._send(f'<{zone}MU{int(mute):02}')

    async def set_volume(self, zone: int, volume: int) -> None:
        """Set zone volume level."""
        self._validate_zone(zone)
        volume = self._clamp(volume, 0, _NERO_CONFIG['max_volume'])
        await self._send(f'<{zone}VO{volume:02}')

    async def set_treble(self, zone: int, treble: int) -> None:
        """Set zone treble level."""
        self._validate_zone(zone)
        treble = self._clamp(treble, 0, _NERO_CONFIG['max_treble'])
        await self._send(f'<{zone}TR{treble:02}')

    async def set_bass(self, zone: int, bass: int) -> None:
        """Set zone bass level."""
        self._validate_zone(zone)
        bass = self._clamp(bass, 0, _NERO_CONFIG['max_bass'])
        await self._send(f'<{zone}BS{bass:02}')

    async def set_balance(self, zone: int, balance: int) -> None:
        """Set zone balance."""
        self._validate_zone(zone)
        balance = self._clamp(balance, 0, _NERO_CONFIG['max_balance'])
        await self._send(f'<{zone}BL{balance:02}')

    async def set_source(self, zone: int, source: int) -> None:
        """Set zone input source."""
        self._validate_zone(zone)
        if source not in _NERO_CONFIG['sources']:
            raise ValueError(f'Invalid Nero MAX 12 source: {source}')
        await self._send(f'<{zone}CH{source:02}')

    async def all_off(self) -> None:
        """Turn off all zones."""
        for zone in _NERO_CONFIG['zones']:
            await self.set_power(zone, False)

    async def restore_zone(self, status: dict[str, Any]) -> None:
        """Restore a zone from snapshot data."""
        zone = int(status['zone'])
        restore_steps: tuple[tuple[str, Callable[[int, Any], Any]], ...] = (
            ('power', self.set_power),
            ('source', self.set_source),
            ('volume', self.set_volume),
            ('mute', self.set_mute),
            ('bass', self.set_bass),
            ('balance', self.set_balance),
            ('treble', self.set_treble),
        )
        for key, setter in restore_steps:
            if key in status:
                await setter(zone, status[key])
