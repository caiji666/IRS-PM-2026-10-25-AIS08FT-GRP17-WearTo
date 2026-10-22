"""Exercise the shipped browser handlers when a changed city finishes ranking."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_completed_refinement_switches_inspiration_and_top3_to_the_same_run():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the browser-handler regression")
    source = Path("travel_agent/web/static/app.js").read_text()
    names = ("applyTop3", "loadResult", "pollStudent5Refinement")
    handlers = "\n".join(
        next(
            line
            for line in source.splitlines()
            if line.startswith((f"function {name}(", f"async function {name}("))
        )
        for name in names
    )
    harness = r"""
const assert = require('node:assert/strict');
const nodes = new Map();
const $ = name => {
  if (!nodes.has(name)) nodes.set(name, {textContent:'',innerHTML:'',close(){}});
  return nodes.get(name);
};
const state = {result:{id:'old-seoul',references:[{image:'old.jpg'}]},
  ref:8,outfits:[{outfit_id:'old'}],messages:[],refinementJob:'job'};
const calls=[];
const data={id:'new-beijing',trip:{destination:'Beijing',start_date:'2026-10-25',
  end_date:'2026-10-28'},references:[{image:'beijing.jpg'}],count:20,owned_count:21};
const ranked={id:'session',run_id:'new-beijing',top3:[{outfit_id:'new1'},
  {outfit_id:'new2'},{outfit_id:'new3'}]};
async function api(url){calls.push(url);return url.startsWith('/api/job/')?
  {status:'completed',result:'new-beijing',student5:ranked}:data;}
const sessionStorage={setItem(){}};
const document={querySelector(){return {scrollIntoView(){}};}};
const text=x=>x;
const dateText=x=>x;
let displayedImage;
function renderReference(){displayedImage=state.result.references[state.ref].image;}
function renderProducts(){}
function resetPreview(){}
function setTop3Loading(){}
function renderOutfitTabs(){}
function renderOutfitProducts(){}
function renderChat(){}
function setRefinementLocked(){}
function toast(message){throw new Error(message);}
function showChatNotice(message){throw new Error(message);}
function progress(){}
async function loadTop3(){throw new Error('Must reuse this run\'s ranked outfits');}
"""
    assertions = r"""
(async()=>{
 await pollStudent5Refinement();
 assert.equal(state.result.id,'new-beijing');
 assert.equal(displayedImage,'beijing.jpg');
 assert.equal(state.ref,0);
 assert.equal(state.outfits.length,3);
 assert.equal(state.outfits[0].outfit_id,'new1');
 assert.match($('#heroTitle').innerHTML,/Beijing/);
 assert.match($('#heroSeason').textContent,/BEIJING/);
 assert.deepEqual(calls,['/api/job/job','/api/result/new-beijing']);
 // A saved journey should also identify its actual destination.
 await loadResult('new-beijing',false,ranked);
 assert.match($('#heroTitle').innerHTML,/Beijing/);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    completed = subprocess.run(
        [node, "-e", harness + "\n" + handlers + "\n" + assertions],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, json.dumps(
        {"stdout": completed.stdout, "stderr": completed.stderr}
    )
