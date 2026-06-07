# Human validation sheet — F1-F9 taxonomy verification

Total samples: **65** (stratified: up to 12 per bucket from 6 buckets).

## Instructions

For each sample below, read the observation + predicted edge + GT hint, and assign a label from the same vocabulary as the subagent. Then save the file and run `experiments/exp24_compute_kappa.py` to compute Cohen's kappa between subagent and human labels.

Annotate by filling in the `Human label: _____` line. You can use any of the six bucket labels listed below each sample, or `UNCERTAIN`.

---

## Sample 000 — game = lostpig

**Predicted bucket (subagent):** `CORRECT`

- step_num: 4
- walkthrough action: `southwest`
- predicted edge: `'fountain room' --[southwest]--> 'table room'`
- observation: `'Pig look happy that Grunk going away.  Table Room This room look like it maybe for eating or for play cards or for maybe for just sitting and talking. That because there big table and one chair here.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 001 — game = night

**Predicted bucket (subagent):** `CORRECT`

- step_num: 6
- walkthrough action: `down`
- predicted edge: `'stairwell (third floor)' --[down]--> 'stairwell (second floor)'`
- observation: `"Stairwell (Second Floor) You're in the north stairwell.  Stairs lead up and down.  There is a door to the east."`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 002 — game = ballyhoo

**Predicted bucket (subagent):** `CORRECT`

- step_num: 7
- walkthrough action: `north`
- predicted edge: `'connection' --[north]--> 'in the wings'`
- observation: `'In the Wings The big top can be entered to the north and exited to the south.   To the northeast, the grandstand has been retracted slightly, revealing a passage.  A roustabout who is wearing a pair o'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 003 — game = awaken

**Predicted bucket (subagent):** `CORRECT`

- step_num: 2
- walkthrough action: `southeast`
- predicted edge: `'west side of the church' --[southeast]--> 'front of the church'`
- observation: `'Front of the Church Two massive oak trees which flank the cracked sidewalk dominate the front of the church. The facade of the church itself rises beyond them, its steeple silhouetted against the sky.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 004 — game = zork2

**Predicted bucket (subagent):** `CORRECT`

- step_num: 8
- walkthrough action: `southwest`
- predicted edge: `'path near stream' --[southwest]--> 'carousel room'`
- observation: `'Carousel Room You are in a large circular room whose high ceiling is lost in gloom. Eight identical passages leave the room. A loud whirring sound comes from all around, and you feel sort of disorient'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 005 — game = cutthroat

**Predicted bucket (subagent):** `CORRECT`

- step_num: 12
- walkthrough action: `south`
- predicted edge: `'vacant lot' --[south]--> 'back alley (south to vacant lot)'`
- observation: `'Back Alley You are in an east/west alley. To the north is a vacant lot, and an overgrown field lies to the south.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 006 — game = night

**Predicted bucket (subagent):** `CORRECT`

- step_num: 9
- walkthrough action: `south`
- predicted edge: `'hall (1st floor, north end of north/south hall)' --[south]--> 'hall (1st floor, middle of north/south hall)'`
- observation: `"Hall You're in the middle of a long north/south hall.  You can go east here as well as north or south."`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 007 — game = reverb

**Predicted bucket (subagent):** `CORRECT`

- step_num: 12
- walkthrough action: `east`
- predicted edge: `'street, near courthouse' --[east]--> 'street, by pizza parlor'`
- observation: `'Street, by Pizza Parlor'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 008 — game = plundered

**Predicted bucket (subagent):** `CORRECT`

- step_num: 6
- walkthrough action: `south`
- predicted edge: `"crew's quarters" --[south]--> 'hold'`
- observation: `'Hold    You gradually notice a familiar, worrying smell, sniff, and identify it as smoke. A tiny glow of fire creeps across a stretch of floor -- inside the cage full of ammunition.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 009 — game = planetfall

**Predicted bucket (subagent):** `CORRECT`

- step_num: 8
- walkthrough action: `east`
- predicted edge: `'rec corridor' --[east]--> 'mess corridor'`
- observation: `'Mess Corridor This is a wide, east-west hallway with a large portal to the south. A small door to the north is closed and hooked with a simple steel padlock which is also closed.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 010 — game = dragon

**Predicted bucket (subagent):** `CORRECT`

- step_num: 3
- walkthrough action: `up`
- predicted edge: `'inn cellar' --[up]--> 'inside the inn'`
- observation: `'Inside the Inn There are a few patches of mould on the walls and the floorboards creak slightly, but the building seems safe enough. As you move you disturb a thin layer of dust. There are some stone'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 011 — game = yomomma

**Predicted bucket (subagent):** `CORRECT`

- step_num: 3
- walkthrough action: `west`
- predicted edge: `'entrance (southern side)' --[west]--> 'arcade corner (sw corner)'`
- observation: `'(In addition to using the compass directions you can move around the club by simply typing the name of the location where you want to go, or a name of a person in the location. You can also see the li'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 012 — game = lurking

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 11
- walkthrough action: `down`
- predicted edge: `'basement' --[down]--> 'temporary basement'`
- observation: `'Temporary Basement'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 013 — game = night

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 19
- walkthrough action: `east`
- predicted edge: `'hall (2nd floor, middle of north/south hall)' --[east]--> 'maze of twisty passages (stop 1)'`
- observation: `'Maze of Twisty Passages You are in a maze of twisty little passages, all alike.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 014 — game = inhumane

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 7
- walkthrough action: `west`
- predicted edge: `'round room' --[west]--> 't-intersection (east/west/south, east to round room)'`
- observation: `'T-Intersection The east/west corridor branches to the south here.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 015 — game = yomomma

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 10
- walkthrough action: `north`
- predicted edge: `'dance floor (center of the club)' --[north]--> 'vip lounge (nw corner)'`
- observation: `"VIP lounge (NW corner) This area is reserved to those with VIP passes. There are cushy red sofas to sit on and you don't have to worry about drunken idiots crashing your table.  A guard is making sure"`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 016 — game = temple

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 12
- walkthrough action: `west`
- predicted edge: `'road (north/south)' --[west]--> 'crossroads'`
- observation: `'Crossroads The road to the public square, way back east, forks into a north-south road here. To the west looms yet another of the stone buildings.  Charles comes walking behind you.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 017 — game = cutthroat

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 16
- walkthrough action: `east`
- predicted edge: `'back alley (east end of east/west alley)' --[east]--> 'back alley (behind outfitters international)'`
- observation: `'Back Alley'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 018 — game = karn

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 14
- walkthrough action: `southwest`
- predicted edge: `'mountain trail (carved steps to the south)' --[southwest]--> 'mountain trail (cleft to northwest, down southwest, down east)'`
- observation: `'Mountain Trail  To the south some steps are carved into the mountain.  They lead upwards out of sight.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 019 — game = night

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 14
- walkthrough action: `north`
- predicted edge: `'hall (1st floor, north end of north/south hall)' --[north]--> 'hall (1st floor, middle of north/south hall)'`
- observation: `'Hall  To one side, a water fountain hums quietly.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 020 — game = yomomma

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 12
- walkthrough action: `southeast`
- predicted edge: `'bar (western side)' --[southeast]--> 'dance floor (center of the club)'`
- observation: `"Dance floor (center of the club) This is the center of the Compass Club. Most people don't orient themselves in relation to compass directions, but here it's more than natural. A huge compass rose is"`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 021 — game = wishbringer

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 4
- walkthrough action: `north`
- predicted edge: `'open grave' --[north]--> 'spooky copse'`
- observation: `"Spooky Copse A copse of willow trees makes this part of the cemetery look really spooky. Narrow lanes wander south and west.  There's an open grave nearby, freshly dug, with a tombstone erected next t"`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 022 — game = lurking

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 10
- walkthrough action: `up`
- predicted edge: `'temporary basement' --[up]--> 'basement'`
- observation: `'It is pitch black.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 023 — game = karn

**Predicted bucket (subagent):** `WRONG_DIRECTION`

- step_num: 15
- walkthrough action: `west`
- predicted edge: `'mountain trail (cleft to northwest, down southwest, down east)' --[west]--> 'mountain trail (up west, down east)'`
- observation: `'Mountain Trail'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 024 — game = zenon

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 6
- walkthrough action: `west`
- predicted edge: `'corridor(east old cell and stretches further west)' --[west]--> 'corridor(south into a room and west to old cell)'`
- GT destination from same (src, dir): `'corridor(east to old cell and north to hall)'`
- observation: `'Corridor A section of corridor that stretches east back toward your old cell and north into some sort of hall. It also leads south to another room.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 025 — game = cutthroat

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 35
- walkthrough action: `north`
- predicted edge: `'vacant lot' --[north]--> 'wharf road (north to mcginty salvage)'`
- GT destination from same (src, dir): `'wharf road (north to outfitters international former warehouse)'`
- observation: `'Wharf Road'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 026 — game = lostpig

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 6
- walkthrough action: `east`
- predicted edge: `'table room' --[east]--> 'shelf room'`
- GT destination from same (src, dir): `'gnome room'`
- observation: `'Shelf Room There lots of shelfs in this room. Them on every wall. This room probably have lots and lots of thing in it before. But shelfs all empty now. Grunk not see any thing there at all. Maybe Gru'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 027 — game = murdac

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 28
- walkthrough action: `east`
- predicted edge: `'high tunnel (east/west, path to west)' --[east]--> 'wooden plank in east/west tunnel'`
- GT destination from same (src, dir): `'high tunnel (east/west, alcove off to north)'`
- observation: `'You are in the east-west tunnel by the alcove, with the wiring to your east. There is a plank lying across the wires'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 028 — game = gold

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 14
- walkthrough action: `north`
- predicted edge: `'in the garden' --[north]--> 'in the pantry'`
- GT destination from same (src, dir): `'in a small meadow'`
- observation: `"In the pantry I'm in the pantry, a small claustrophobic room which smells faintly of cinammon, honey and the mouse droppings that are liberally sprinkled across the tiled floor. There is an upright fr"`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 029 — game = detective

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 14
- walkthrough action: `north`
- predicted edge: `'hallway (15th floor, room 19-22)' --[north]--> 'outside (holiday inn to north, doughnut king to east, the wall to west)'`
- GT destination from same (src, dir): `'hallway (sauna to west, pool a to east)'`
- observation: `'<< Outside >> You pass the guard. He nods at you. You are now outside standing on the street. You can go north and east, your choice. To the north is more of the street, and to the east is a video sto'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 030 — game = yomomma

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 13
- walkthrough action: `south`
- predicted edge: `'dance floor (center of the club)' --[south]--> 'hall of fame (eastern side)'`
- GT destination from same (src, dir): `'entrance (southern side)'`
- observation: `"Hall of Fame (eastern side) Just being here makes your heart beat faster. The walls are adorned by portraits of past champions, each a part of great history.  Ralph, one of Gus's goons, is sitting her"`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 031 — game = anchor

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 4
- walkthrough action: `west`
- predicted edge: `'file room' --[west]--> 'outside the real estate office'`
- GT destination from same (src, dir): `'office'`
- observation: `'(opening the real estate office door first) The rain is still coming down, so you open your umbrella.  Outside the Real Estate Office A grim little cul-de-sac, tucked away in a corner of the claustrop'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 032 — game = detective

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 13
- walkthrough action: `east`
- predicted edge: `'bedroom' --[east]--> 'hallway (15th floor, room 19-22)'`
- GT destination from same (src, dir): `"hallway (mayer's house, east/west intersect, exit to north)"`
- observation: `'<< Hallway >> You are still in the hallway. You can go north to where there is a police officer who will let you outside, or you can go east or west.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 033 — game = sherlock

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 15
- walkthrough action: `east`
- predicted edge: `'marylebone road' --[east]--> 'parliament square'`
- GT destination from same (src, dir): `'tottenham court road'`
- observation: `'> Saturday  6:00:00 a.m. In the distance, you hear Big Ben strike the hour. [Press any key to continue.]'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 034 — game = dragon

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 26
- walkthrough action: `north`
- predicted edge: `'wasteland by a castle' --[north]--> 'forbidding castle'`
- GT destination from same (src, dir): `'castle forge'`
- observation: `'Forbidding Castle The hulk of a massive, stone castle towers into the sky above you. The walls are covered in grey lichen and moss, and seem to have stood here since the beginning of time itself. Set'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 035 — game = cutthroat

**Predicted bucket (subagent):** `WRONG_DST`

- step_num: 33
- walkthrough action: `west`
- predicted edge: `'back alley (behind outfitters international)' --[west]--> 'back alley (west end of east/west alley)'`
- GT destination from same (src, dir): `'back alley (south to vacant lot)'`
- observation: `'Back Alley'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 036 — game = sherlock

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 16
- walkthrough action: `east`
- predicted edge: `'parliament square' --[east]--> 'scotland yard'`
- observation: `'> Scotland Yard  You walk up the steps and into the building. Inside, it looks just like any other government building - drab and worn. The door to the street is to the west, and you see some steps le'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 037 — game = night

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 20
- walkthrough action: `west`
- predicted edge: `'maze of twisty passages (stop 1)' --[west]--> 'maze of twisty passages (stop 5)'`
- observation: `'Maze of Twisty Passages'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 038 — game = spellbrkr

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 5
- walkthrough action: `south`
- predicted edge: `'ruins room' --[south]--> 'cliff middle'`
- observation: `'As you leave, the "1" cube reappears in your hand.  Cliff Middle A narrow ledge, barely wide enough to stand on, interrupts the cliff here. There is a dirty scroll here.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 039 — game = zork1

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 20
- walkthrough action: `east`
- predicted edge: `'temple' --[east]--> 'forest path'`
- observation: `'Forest Path'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 040 — game = advent

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 33
- walkthrough action: `east`
- predicted edge: `'misty cavern' --[east]--> 'plover room'`
- observation: `"Plover Room You're in a small chamber lit by an eerie green light. An extremely narrow tunnel exits to the west. A dark corridor leads northeast.  There is an emerald here the size of a plover's egg!"`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 041 — game = awaken

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 6
- walkthrough action: `north`
- predicted edge: `'east side of the church' --[north]--> 'church porch'`
- observation: `'Church Porch This wide, covered porch spans the width of the church front. A white picket railing surrounds the porch on three sides, except to the south where a short flight of steps leads down to th'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 042 — game = huntdark

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 2
- walkthrough action: `up`
- predicted edge: `'cramped cavern' --[up]--> 'small cave'`
- observation: `'Small Cave This is a small cave. Ropy pillars rise from floor to roof, and water trickles audibly, somewhere nearby. The floor is crusted with wet sand. A low tunnel leads ahead, a path goes right, an'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 043 — game = trinity

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 5
- walkthrough action: `east`
- predicted edge: `'black lion gate' --[east]--> 'wading'`
- observation: `"You wade into the cool, dark water.  Wading  You're standing knee-deep in the Long Water, not far from the western shore. Looking east, you can see a white door hovering just above the surface.  Swans"`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 044 — game = inhumane

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 18
- walkthrough action: `north`
- predicted edge: `'end of glass hall' --[north]--> 'landing'`
- observation: `'Landing This is a landing on the stairway. You can go down to the north or up to the south. A passage leads east.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 045 — game = huntdark

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 1
- walkthrough action: `up`
- predicted edge: `'tight crawl' --[up]--> 'cramped cavern'`
- observation: `'Cramped Cavern You are in a cramped cavern, with a high roof. Stalactites hang low overhead, and you can hear water rushing in the distance. The ground is worn into polished flowing curves. A pit lead'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 046 — game = awaken

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 14
- walkthrough action: `down`
- predicted edge: `'vestibule' --[down]--> 'steeple chamber'`
- observation: `'Steeple Chamber  A frayed but heavy rope hangs from somewhere overhead. There is a tree branch hanging from the end of the rope.  The heavy wooden door to the north is closed.  The top of the broken s'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 047 — game = curses

**Predicted bucket (subagent):** `SPURIOUS_PAIR`

- step_num: 5
- walkthrough action: `west`
- predicted edge: `'airing cupboard' --[west]--> 'potting room'`
- observation: `'Potting Room This light room is full of pot plants, flowers, seeds, ornamental trowels and other miscellaneous garden implements.  A pair of yellow rubber gloves hangs from a hook on one wall.  Aunt J'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 048 — game = spirit

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 5
- walkthrough action: `north`
- predicted edge: `'main hall' --[north]--> 'monastery chapel'`
- observation: `'Monastery Chapel You are in the Monastery Chapel, the site of the recent appearance of the legendary Anabais. Signs of the recent visit include the slightly scorched walls, and the reek of brimstone.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 049 — game = cutthroat

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 28
- walkthrough action: `south`
- predicted edge: `"mariners' trust" --[south]--> 'shore road'`
- observation: `'Shore Road'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 050 — game = afflicted

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 10
- walkthrough action: `east`
- predicted edge: `'refrigerator' --[east]--> 'darkness'`
- observation: `'Darkness'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 051 — game = balances

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 10
- walkthrough action: `east`
- predicted edge: `'edge of chasm' --[east]--> 'outside temple'`
- observation: `'Track, outside Temple This is the end of a long track winding through desolate hills, which runs back west up to the ridge.  You stand outside an enormous temple in the shape of a perfect, featureless'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 052 — game = pentari

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 7
- walkthrough action: `north`
- predicted edge: `'main hall' --[north]--> 'by the fireplace'`
- observation: `'Main Hall, by the Fireplace You are at the northern end of the main hall taking in the panorama of what was once a majestic area where perhaps treaties were signed, strategies laid and plans foiled.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 053 — game = ludicorp

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 34
- walkthrough action: `north`
- predicted edge: `'storeroom' --[north]--> 'west end'`
- observation: `'Long Corridor, West End You are at the west end of a very long east-west corridor. Doors lead north and south.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 054 — game = reverb

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 8
- walkthrough action: `west`
- predicted edge: `'law office' --[west]--> 'second floor'`
- observation: `'Office Building, Second Floor'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 055 — game = detective

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 11
- walkthrough action: `north`
- predicted edge: `"hallway (mayer's house, east/west intersect, exit to north)" --[north]--> 'outside (restaurant to north, mayer home to east)'`
- observation: `'<< Hallway >> You are still in the hallway. You can go north to where there is a police officer who will let you outside, or you can go east or west.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 056 — game = afflicted

**Predicted bucket (subagent):** `HALLUCINATED_DST`

- step_num: 12
- walkthrough action: `south`
- predicted edge: `'dish room' --[south]--> 'darkness'`
- observation: `'Darkness'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 057 — game = spirit

**Predicted bucket (subagent):** `HALLUCINATED_SRC`

- step_num: 6
- walkthrough action: `south`
- predicted edge: `'monastery chapel' --[south]--> 'main hall'`
- observation: `'Main Hall'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 058 — game = cutthroat

**Predicted bucket (subagent):** `HALLUCINATED_SRC`

- step_num: 29
- walkthrough action: `northeast`
- predicted edge: `'shore road' --[northeast]--> 'ocean road (halfway of north/south road)'`
- observation: `'Ocean Road'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 059 — game = detective

**Predicted bucket (subagent):** `HALLUCINATED_SRC`

- step_num: 12
- walkthrough action: `west`
- predicted edge: `'outside (restaurant to north, mayer home to east)' --[west]--> 'bedroom'`
- observation: `'<< Bedroom >> You are in the bedroom. You noticed that there was a guard guarding the stairs to the 3rd story, because there is remodelling going on there. You see nothing of importance. Go east.  [Yo'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 060 — game = afflicted

**Predicted bucket (subagent):** `HALLUCINATED_SRC`

- step_num: 11
- walkthrough action: `north`
- predicted edge: `'darkness' --[north]--> 'dish room'`
- observation: `'Dish Room On entering the dish room you are overwhelmed by the astringent smell of fresh bleach.  You would welcome this as evidence of sanitation, except that in such high concentrations bleach is ac'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 061 — game = pentari

**Predicted bucket (subagent):** `HALLUCINATED_SRC`

- step_num: 8
- walkthrough action: `south`
- predicted edge: `'by the fireplace' --[south]--> 'main hall'`
- observation: `'Main Hall'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 062 — game = reverb

**Predicted bucket (subagent):** `HALLUCINATED_SRC`

- step_num: 9
- walkthrough action: `east`
- predicted edge: `'second floor' --[east]--> "mayor's office"`
- observation: `"Mayor's Office You are in the mayor's office. To the east is a window overlooking a second-story drop. There is a glass box mounted on the north wall.  You can see a file cabinet (which is closed) her"`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 063 — game = afflicted

**Predicted bucket (subagent):** `HALLUCINATED_SRC`

- step_num: 13
- walkthrough action: `south`
- predicted edge: `'darkness' --[south]--> 'galley kitchen'`
- observation: `'Galley Kitchen'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

## Sample 064 — game = ludicorp

**Predicted bucket (subagent):** `HALLUCINATED_SRC`

- step_num: 35
- walkthrough action: `north`
- predicted edge: `'west end' --[north]--> 'north-south corridoor'`
- observation: `'North-South Corridoor You are in a narrow corridoor, running north to south.'`

**Human label:** _____  (choose one):
- `CORRECT` — predicted edge exactly matches GT
- `WRONG_DIRECTION` — (src, dst) right, direction wrong
- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)
- `SPURIOUS_PAIR` — both nodes real but no GT edge exists
- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)
- `HALLUCINATED_SRC` — src is not a GT room
- `UNCERTAIN` — observation too ambiguous to decide

**Notes:** 

---

