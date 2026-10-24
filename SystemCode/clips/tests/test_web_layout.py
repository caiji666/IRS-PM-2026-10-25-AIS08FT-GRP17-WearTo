"""Discovery stays in two panels until actual recommendations are available."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_empty_history_results_and_new_journey_have_consistent_layout():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for frontend state regression")
    source = Path("travel_agent/web/static/app.js").read_text()
    names = (
        "updateJourneyLayout",
        "applyTop3",
        "loadTop3",
        "loadResult",
        "history",
        "startNewJourney",
    )
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
const nodes=new Map();
const $=key=>{if(!nodes.has(key))nodes.set(key,{hidden:false,value:'',innerHTML:'',textContent:'',
 classList:{toggle(name,on){this[name]=on}},focus(){}});return nodes.get(key)};
const editLink={hidden:false};
const document={querySelector(){return editLink}};
const state={outfits:[],result:null,messages:[],request:{},choiceSelection:new Set(),busy:false};
const storage=new Map([['away-saved','preserve'],['away-chat-id','old'],
 ['student5-session-id','old']]);
const sessionStorage={removeItem:key=>storage.delete(key),
 setItem:(key,value)=>storage.set(key,value)};
let neutralRenders=0;
function renderReference(){if(!state.result)neutralRenders++;}
function renderOutfitTabs(){updateJourneyLayout()}
function renderOutfitProducts(){}
function renderChat(){updateJourneyLayout()}
function renderProducts(){}
function resetPreview(){}
function setTop3Loading(){}
function showChatNotice(){}
function dateText(value){return value;}
function stampText(value){return value;}
function text(value){return value;}
function toast(error){throw new Error(error)}
let pendingHistory=null, entries=[];
async function api(path){
 if(path==='/api/history')return pendingHistory||entries;
 if(path.startsWith('/api/result/'))return {id:'saved',references:[{image:'saved.jpg'}],
  trip:{destination:'Seoul'},owned_count:0};
 if(path==='/api/student5/start')return {id:'ranked',
  top3:[{outfit_id:'one'},{outfit_id:'two'},{outfit_id:'three'}]};
 throw new Error(path);
}
"""
    assertions = r"""
(async()=>{
 await history();
 assert.equal($('#collection').hidden,true);
 assert.equal($('#journey').classList['is-discovery'],true);
 assert.equal(editLink.hidden,true);
 assert(neutralRenders>0);
 // Typing and collecting facts alone do not open an empty product panel.
 state.messages.push({role:'user',content:'Seoul'});renderChat();
 assert.equal($('#collection').hidden,true);
 entries=[{id:'saved'}];await history();
 assert.equal(state.outfits.length,3);
 assert.equal($('#collection').hidden,false);
 assert.equal(editLink.hidden,false);
 assert.equal($('#journey').classList['is-discovery'],false);
 assert.equal($('.chat-panel').hidden,true);
 assert.equal($('#journey').classList['is-edit-focus'],true);
 // Adjust requirements explicitly opens the third panel; closing it preserves results.
 state.chatExpanded=true;renderChat();
 assert.equal($('.chat-panel').hidden,false);
 assert.equal($('#collection').hidden,false);
 assert.equal($('#journey').classList['is-edit-focus'],false);
 applyTop3({id:'next',top3:[{outfit_id:'new'}]});
 assert.equal($('.chat-panel').hidden,true);
 assert.equal(state.chatExpanded,false);
 startNewJourney();
 assert.equal($('#collection').hidden,true);
 assert.equal(state.result,null);
 assert.equal(state.student5,null);
 assert.equal(storage.get('away-saved'),'preserve');
 assert.equal(state.messages.length,0);
 // An in-flight history request cannot reopen the previous edit after reset.
 let resolveHistory;
 pendingHistory=new Promise(resolve=>resolveHistory=resolve);
 const stale=history();startNewJourney();resolveHistory([{id:'saved'}]);await stale;
 assert.equal($('#collection').hidden,true);
 assert.equal(state.result,null);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(
        [node, "-e", harness + "\n" + handlers + "\n" + assertions],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
