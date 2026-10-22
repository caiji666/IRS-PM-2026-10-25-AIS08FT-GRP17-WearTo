"""New-trip dialogue must remain independent of a displayed Student 5 session."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario", ["initial", "new_journey", "offer", "refine"])
def test_browser_chat_uses_the_intended_controller(scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser routing regression")
    source = Path("travel_agent/web/static/app.js").read_text()
    handlers = "\n".join(
        next(
            line
            for line in source.splitlines()
            if line.startswith((f"function {name}(", f"async function {name}("))
        )
        for name in ("startNewJourney", "submitChatMessage", "focusRequirementChat")
    )
    harness = r"""
const assert=require('node:assert/strict');
const state={chatMode:'trip',student5:'old-session',outfits:[{outfit_id:'old'}],
 job:null,messages:[],request:{},review:null,offer:null,busy:false,
 choiceSelection:new Set(),pendingRefinement:null,refinementJob:null};
const storage=new Map();
const sessionStorage={removeItem(k){storage.delete(k)},setItem(k,v){storage.set(k,v)}};
const elements=new Map();
const $=key=>{if(!elements.has(key))elements.set(key,{value:'',style:{},textContent:'',focus(){}});
 return elements.get(key);};
const document={querySelector(){return {scrollIntoView(){}};}};
function showChatNotice(){}
function renderChat(){}
function setBusy(value){state.busy=value;}
function resetPreview(){}
function renderOutfitTabs(){}
function renderOutfitProducts(){}
function showRefinementPlan(){}
function poll(){}
const calls=[];
async function api(path,body){calls.push({path,body});return path==='/api/chat'?
 {id:'new-trip-job'}:{top3:[{outfit_id:'updated'}]};}
"""
    scenarios = {
        "initial": "await submitChatMessage('I want to visit Tokyo.');",
        "new_journey": "state.chatMode='refine';state.job='old-trip';startNewJourney();"
        "assert.equal(state.job,null);await submitChatMessage('I want to visit Tokyo.');",
        "offer": "state.chatMode='refine';state.job='new-trip';"
        "await submitChatMessage('Museums','optional-activities');",
        "refine": "focusRequirementChat();await submitChatMessage('Change my budget to 200.');",
    }
    expected = "/api/student5/refine" if scenario == "refine" else "/api/chat"
    expected_offer = '"optional-activities"' if scenario == "offer" else "undefined"
    assertions = f"""
(async()=>{{
 {scenarios[scenario]}
 assert.equal(calls.length,1);
 assert.equal(calls[0].path,{expected!r});
 if(calls[0].path==='/api/chat'){{
  assert.equal(state.job,'new-trip-job');
  assert.equal(calls[0].body.offer_key,{expected_offer});
 }}
}})().catch(error=>{{console.error(error);process.exitCode=1;}});
"""
    result = subprocess.run(
        [node, "-e", harness + "\n" + handlers + "\n" + assertions],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
