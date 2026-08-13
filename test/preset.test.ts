/**
 * Preset registry tests: every shipped preset loads and validates, and the
 * validator rejects malformed data.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { listPresetIds, loadPreset, validatePreset } from '../src/preset.ts';

test('every shipped preset loads with a coherent shape', () => {
  const ids = listPresetIds();
  assert.ok(ids.length >= 6, 'the six launch presets exist');
  for (const id of ids) {
    const preset = loadPreset(id);
    assert.equal(preset.id, id);
    assert.ok(preset.displayName.length > 0, `${id}: displayName`);
    assert.ok(preset.baseUrl.startsWith('http'), `${id}: baseUrl`);
    assert.ok(['key', 'oauth', 'none'].includes(preset.auth.kind), `${id}: auth kind`);
    assert.ok(preset.models.length > 0, `${id}: models`);
    for (const model of preset.models) {
      assert.ok(model.id.length > 0, `${id}: model id`);
    }
    assert.match(preset.verifiedAt ?? '', /^\d{4}-\d{2}-\d{2}$/, `${id}: verifiedAt`);
  }
});

test('the recommended default chain presets all exist', () => {
  for (const id of ['nous-portal', 'opencode-zen', 'deepseek-official', 'kilo']) {
    assert.doesNotThrow(() => loadPreset(id), id);
  }
});

test('deepseek-family presets declare the deepseek quirks', () => {
  const preset = loadPreset('deepseek-official');
  assert.equal(preset.quirks?.usage, 'deepseek');
  assert.equal(preset.quirks?.thinkingField, true);
});

test('deepseek-official carries pricing for cost estimates', () => {
  const preset = loadPreset('deepseek-official');
  assert.ok(preset.pricing !== undefined);
  assert.equal(preset.pricing!.inputPerMTokens, 0.14);
  assert.equal(preset.pricing!.outputPerMTokens, 0.28);
});

test('custom preset defaults to no auth', () => {
  const preset = loadPreset('custom');
  assert.equal(preset.auth.kind, 'none');
});

test('validatePreset rejects malformed documents', () => {
  assert.throws(() => validatePreset({ displayName: 'x', baseUrl: 'http://x', auth: { kind: 'nope' }, models: [] }, 'bad'), /auth\.kind/);
  assert.throws(() => validatePreset({ displayName: 'x', baseUrl: 'http://x', auth: { kind: 'key' }, models: [{ id: '' }] }, 'bad'), /model without an id/);
  assert.throws(() => validatePreset({ displayName: '', baseUrl: 'http://x', auth: { kind: 'key' }, models: [] }, 'bad'), /displayName/);
  assert.throws(() => validatePreset({ displayName: 'x', baseUrl: 'http://x', auth: { kind: 'key' }, models: [], free: { tier: 'free-ish' } }, 'bad'), /free\.tier/);
  assert.throws(() => validatePreset({ displayName: 'x', baseUrl: 'http://x', auth: { kind: 'key' }, models: [], quirks: { usage: 'weird' } }, 'bad'), /quirks\.usage/);
  assert.throws(() => validatePreset({ displayName: 'x', baseUrl: 'http://x', auth: { kind: 'key' }, models: [], headers: { 'X-Test': 42 } }, 'bad'), /headers\.X-Test/);
});

test('loadPreset throws on an unknown id', () => {
  assert.throws(() => loadPreset('does-not-exist'), /not found or not valid JSON/);
});
