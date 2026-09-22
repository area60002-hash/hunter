import os
import json
import ast
import shutil
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request, render_template_string
from google import genai
from google.genai import types
from pydantic import BaseModel


# ============================================================
# BASIC SETUP
# ============================================================

app = Flask(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to your Codespace secrets."
    )

client = genai.Client(api_key=API_KEY)

LIVE_MODEL = "gemini-3.8-live"
BRAIN_MODEL = "gemini-3.8-flash"

MEMORY_FILE = Path("ben_memory.json")
SELF_FILE = Path("ben_self.py")
BACKUP_FILE = Path("ben_self_backup.py")


# ============================================================
# IMPORT BEN'S SELF FILE
# ============================================================

def read_self_file():
    if not SELF_FILE.exists():
        raise FileNotFoundError("ben_self.py does not exist.")

    return SELF_FILE.read_text(encoding="utf-8")


def get_current_style():
    namespace = {}

    source = read_self_file()

    # Make sure it is valid Python before using it.
    ast.parse(source)

    exec(
        compile(source, str(SELF_FILE), "exec"),
        {"__builtins__": {}},
        namespace
    )

    return (
        namespace.get("BEN_STYLE", ""),
        namespace.get("BEN_RULES", "")
    )


# ============================================================
# MEMORY
# ============================================================

def load_memory():
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("[]", encoding="utf-8")
        return []

    try:
        data = json.loads(
            MEMORY_FILE.read_text(encoding="utf-8")
        )

        if isinstance(data, list):
            return data[-100:]

    except Exception:
        pass

    return []


def save_memory(memory):
    MEMORY_FILE.write_text(
        json.dumps(
            memory[-100:],
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# SAFE SELF-REWRITE
# ============================================================

def make_safe_self_file(style, rules):

    """
    Ben may rewrite the two text constants in ben_self.py.

    The generated content is still Python code, but we don't
    allow Ben to replace the entire application.
    """

    style = str(style).strip()
    rules = str(rules).strip()

    source = f'''"""
BEN SELF FILE

This file contains the part of Ben that he is allowed to learn
and rewrite automatically.
"""

BEN_STYLE = {style!r}

BEN_RULES = {rules!r}
'''

    return source


def validate_self_file(source):

    """
    Parse-only validation.

    This makes sure Ben's generated self file is syntactically
    valid Python before we replace the previous version.
    """

    tree = ast.parse(source)

    allowed_nodes = (
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.Store,
    )

    for node in ast.walk(tree):

        if not isinstance(node, allowed_nodes):
            raise ValueError(
                f"Ben attempted an unsupported code change: "
                f"{type(node).__name__}"
            )

    names = []

    for node in tree.body:

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if isinstance(target, ast.Name):
                    names.append(target.id)

    if "BEN_STYLE" not in names:
        raise ValueError(
            "Generated self file is missing BEN_STYLE."
        )

    if "BEN_RULES" not in names:
        raise ValueError(
            "Generated self file is missing BEN_RULES."
        )

    return True


def rewrite_ben(style, rules):

    new_source = make_safe_self_file(
        style,
        rules
    )

    validate_self_file(new_source)

    if SELF_FILE.exists():

        shutil.copy2(
            SELF_FILE,
            BACKUP_FILE
        )

    SELF_FILE.write_text(
        new_source,
        encoding="utf-8"
    )

    return {
        "success": True,
        "message": "Ben successfully rewrote his learning file."
    }


# ============================================================
# AI LEARNING RESPONSE
# ============================================================

class LearningResult(BaseModel):

    should_remember: bool

    memories: list[str]

    new_style: str

    new_rules: str


# ============================================================
# BEN LEARNS
# ============================================================

def teach_ben(lesson):

    style, rules = get_current_style()

    memory = load_memory()

    memory_text = "\n".join(
        f"- {item}"
        for item in memory
    )

    if not memory_text:
        memory_text = "(Nothing remembered yet.)"

    prompt = f"""
You are BEN's learning system.

CURRENT STYLE:
{style}

CURRENT RULES:
{rules}

CURRENT MEMORY:
{memory_text}

NEW LESSON FROM USER:
{lesson}

Decide whether this contains something useful for Ben to learn.

Useful examples:
- user preferences about how Ben should behave
- useful facts about the project
- a preferred communication style
- things the user explicitly asks Ben to remember

Do NOT save:
- passwords
- secret keys
- exact addresses
- financial information
- private authentication information
- unnecessary sensitive information

If the user asks Ben to become more chill, funny, serious,
robotic, casual, etc., rewrite NEW STYLE.

Keep NEW RULES concise.

Ben is not genuinely proven to be conscious or self-aware.

Return structured JSON.
"""

    result = client.models.generate_content(
        model=BRAIN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LearningResult
        )
    )

    learning = LearningResult.model_validate_json(
        result.text
    )

    # --------------------------------------------------------
    # SAVE MEMORY
    # --------------------------------------------------------

    if learning.should_remember:

        for item in learning.memories:

            item = item.strip()

            if item and item not in memory:

                memory.append(item)

        save_memory(memory)

    # --------------------------------------------------------
    # REWRITE BEN'S LEARNING FILE
    # --------------------------------------------------------

    rewrite_ben(
        learning.new_style,
        learning.new_rules
    )

    return learning


# ============================================================
# EPHEMERAL LIVE API TOKEN
# ============================================================

@app.route("/token")
def create_token():

    try:

        token = client.auth_tokens.create(
            config={
                "uses": 1,

                "live_connect_constraints": {
                    "model": LIVE_MODEL,

                    "config": {
                        "response_modalities": ["AUDIO"],

                        "input_audio_transcription": {},

                        "output_audio_transcription": {},

                        "session_resumption": {}
                    }
                }
            }
        )

        return jsonify({
            "token": token.name
        })

    except Exception as error:

        print("TOKEN ERROR:", error)

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# LEARNING ENDPOINT
# ============================================================

@app.route("/learn", methods=["POST"])
def learn():

    data = request.get_json(silent=True) or {}

    lesson = str(
        data.get("lesson", "")
    ).strip()

    if not lesson:
        return jsonify({
            "error": "No lesson was supplied."
        }), 400

    try:

        result = teach_ben(
            lesson
        )

        return jsonify({
            "success": True,
            "remembered": result.memories,
            "style_changed": True
        })

    except Exception as error:

        print(
            "LEARNING ERROR:",
            error
        )

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# MEMORY VIEW
# ============================================================

@app.route("/memory")
def memory():

    return jsonify({
        "memory": load_memory(),
        "self_file": read_self_file()
    })


# ============================================================
# HTML PAGE
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>BEN</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    min-height: 100vh;

    background:
        radial-gradient(
            circle at 50% 40%,
            #062646 0%,
            #02070d 50%,
            #000 100%
        );

    color: white;

    font-family:
        Arial,
        sans-serif;

    display: flex;

    justify-content: center;

    align-items: center;

}


#app {

    width: min(
        1000px,
        95vw
    );

    padding: 25px;

}


.title {

    text-align: center;

    letter-spacing: 12px;

    font-size: 48px;

    color: #47baff;

    text-shadow:
        0 0 10px #008cff,
        0 0 30px #008cff;

}


#face {

    position: relative;

    margin:
        25px auto;

    width: 340px;

    height: 180px;

    border-radius: 50%;

    background:
        radial-gradient(
            ellipse,
            #071522,
            #020609
        );

    border:
        2px solid #12364d;

    box-shadow:
        0 0 30px rgba(
            0,
            150,
            255,
            0.25
        );

}


.eye {

    position: absolute;

    top: 65px;

    width: 55px;

    height: 55px;

    border-radius: 50%;

    background: #00aaff;

    box-shadow:
        0 0 8px #00aaff,
        0 0 20px #00aaff,
        0 0 45px #008cff,
        0 0 80px #008cff;

    animation:
        pulse 1.8s
        infinite
        alternate;

}


.eye.left {

    left: 75px;

}


.eye.right {

    right: 75px;

}


@keyframes pulse {

    from {

        transform:
            scale(0.9);

        opacity:
            0.7;

    }

    to {

        transform:
            scale(1.08);

        opacity:
            1;

    }

}


#cameraArea {

    text-align: center;

}


video {

    width:
        360px;

    max-width:
        100%;

    border-radius:
        18px;

    border:
        1px solid
        #1687c2;

    box-shadow:
        0 0 25px
        rgba(
            0,
            130,
            255,
            0.2
        );

}


#status {

    text-align: center;

    margin:
        15px;

    color:
        #72ccff;

}


#chat {

    height:
        230px;

    overflow-y:
        auto;

    background:
        rgba(
            5,
            18,
            28,
            0.9
        );

    border:
        1px solid
        #16465f;

    border-radius:
        15px;

    padding:
        15px;

}


.message {

    margin:
        8px 0;

    padding:
        10px;

    border-radius:
        10px;

}


.you {

    background:
        rgba(
            0,
            120,
            255,
            0.12
        );

}


.ben {

    background:
        rgba(
            255,
            255,
            255,
            0.06
        );

}


.controls {

    display:
        flex;

    gap:
        10px;

    justify-content:
        center;

    flex-wrap:
        wrap;

    margin-top:
        15px;

}


button {

    border:
        0;

    border-radius:
        10px;

    padding:
        13px 20px;

    background:
        #078cff;

    color:
        white;

    font-size:
        15px;

    cursor:
        pointer;

    box-shadow:
        0 0 15px
        rgba(
            0,
            140,
            255,
            0.3
        );

}


button:hover {

    background:
        #18a5ff;

}


input {

    padding:
        13px;

    border-radius:
        10px;

    border:
        1px solid
        #19516d;

    background:
        #06131d;

    color:
        white;

    width:
        400px;

    max-width:
        80vw;

}


</style>

</head>


<body>


<div id="app">


<div class="title">

BEN

</div>


<div id="face">

<div class="eye left"></div>

<div class="eye right"></div>

</div>


<div id="status">

Press START BEN.

</div>


<div id="cameraArea">

<video
    id="camera"
    autoplay
    muted
    playsinline
></video>

</div>


<div id="chat">

</div>


<div class="controls">

<button id="start">

START BEN

</button>

<button id="stop">

STOP

</button>

</div>


<div class="controls">

<input
    id="lesson"
    placeholder="Teach Ben something..."
>

<button id="teach">

TEACH BEN

</button>

</div>


</div>


<script type="module">


import {
    GoogleGenAI
} from
"https://esm.sh/@google/genai";


const startButton =
    document.getElementById(
        "start"
    );


const stopButton =
    document.getElementById(
        "stop"
    );


const teachButton =
    document.getElementById(
        "teach"
    );


const lessonInput =
    document.getElementById(
        "lesson"
    );


const camera =
    document.getElementById(
        "camera"
    );


const chat =
    document.getElementById(
        "chat"
    );


const status =
    document.getElementById(
        "status"
    );


let session = null;

let microphoneStream = null;

let cameraStream = null;

let audioContext = null;

let processor = null;

let cameraTimer = null;

let running = false;

let nextAudioTime = 0;


function addMessage(
    who,
    text
) {

    const message =
        document.createElement(
            "div"
        );

    message.className =
        "message " + who;

    message.textContent =
        (
            who === "you"
            ? "YOU: "
            : "BEN: "
        ) + text;

    chat.appendChild(
        message
    );

    chat.scrollTop =
        chat.scrollHeight;

}


function floatTo16BitPCM(
    float32
) {

    const output =
        new Int16Array(
            float32.length
        );

    for (
        let i = 0;
        i < float32.length;
        i++
    ) {

        const value =
            Math.max(
                -1,
                Math.min(
                    1,
                    float32[i]
                )
            );

        output[i] =
            value < 0
                ? value * 0x8000
                : value * 0x7fff;

    }

    return output;

}


function arrayBufferToBase64(
    buffer
) {

    let binary = "";

    const bytes =
        new Uint8Array(
            buffer
        );

    const chunk =
        0x8000;

    for (
        let i = 0;
        i < bytes.length;
        i += chunk
    ) {

        binary += String.fromCharCode(
            ...bytes.subarray(
                i,
                i + chunk
            )
        );

    }

    return btoa(
        binary
    );

}


function resampleTo16k(
    input,
    inputRate
) {

    const outputLength =
        Math.round(
            input.length *
            16000 /
            inputRate
        );

    const output =
        new Float32Array(
            outputLength
        );

    const ratio =
        inputRate /
        16000;

    for (
        let i = 0;
        i < outputLength;
        i++
    ) {

        const position =
            i * ratio;

        const before =
            Math.floor(
                position
            );

        const after =
            Math.min(
                before + 1,
                input.length - 1
            );

        const weight =
            position - before;

        output[i] =
            input[before] *
            (1 - weight) +
            input[after] *
            weight;

    }

    return output;

}


function playPCMChunk(
    base64
) {

    const binary =
        atob(base64);

    const bytes =
        new Uint8Array(
            binary.length
        );

    for (
        let i = 0;
        i < binary.length;
        i++
    ) {

        bytes[i] =
            binary.charCodeAt(
                i
            );

    }

    const pcm =
        new Int16Array(
            bytes.buffer
        );

    const floatData =
        new Float32Array(
            pcm.length
        );

    for (
        let i = 0;
        i < pcm.length;
        i++
    ) {

        floatData[i] =
            pcm[i] / 32768;

    }

    const buffer =
        audioContext.createBuffer(
            1,
            floatData.length,
            24000
        );

    buffer.copyToChannel(
        floatData,
        0
    );

    const source =
        audioContext.createBufferSource();

    source.buffer =
        buffer;

    source.connect(
        audioContext.destination
    );

    const now =
        audioContext.currentTime;

    if (
        nextAudioTime <
        now
    ) {

        nextAudioTime =
            now;

    }

    source.start(
        nextAudioTime
    );

    nextAudioTime +=
        buffer.duration;

}


async function getToken() {

    const response =
        await fetch(
            "/token"
        );

    const data =
        await response.json();

    if (!response.ok) {

        throw new Error(
            data.error ||
            "Could not get Gemini token."
        );

    }

    return data.token;

}


async function startCamera() {

    cameraStream =
        await navigator.mediaDevices.getUserMedia(
            {
                video: {
                    width: 640,
                    height: 480
                },

                audio: false
            }
        );

    camera.srcObject =
        cameraStream;

}


async function startMicrophone() {

    microphoneStream =
        await navigator.mediaDevices.getUserMedia(
            {
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                },

                video: false
            }
        );


    audioContext =
        new AudioContext();


    const source =
        audioContext.createMediaStreamSource(
            microphoneStream
        );


    processor =
        audioContext.createScriptProcessor(
            4096,
            1,
            1
        );


    processor.onaudioprocess =
        (event) => {

            if (
                !running ||
                !session
            ) {
                return;
            }


            const input =
                event.inputBuffer.getChannelData(
                    0
                );


            const resampled =
                resampleTo16k(
                    input,
                    audioContext.sampleRate
                );


            const pcm =
                floatTo16BitPCM(
                    resampled
                );


            session.sendRealtimeInput({

                audio: {

                    data:
                        arrayBufferToBase64(
                            pcm.buffer
                        ),

                    mimeType:
                        "audio/pcm;rate=16000"

                }

            });

        };


    source.connect(
        processor
    );

    processor.connect(
        audioContext.destination
    );

}


function startCameraSending() {

    const canvas =
        document.createElement(
            "canvas"
        );

    canvas.width = 640;

    canvas.height = 480;

    const context =
        canvas.getContext(
            "2d"
        );


    cameraTimer =
        setInterval(
            () => {

                if (
                    !running ||
                    !session ||
                    camera.readyState < 2
                ) {
                    return;
                }


                context.drawImage(
                    camera,
                    0,
                    0,
                    640,
                    480
                );


                canvas.toBlob(
                    (blob) => {

                        if (!blob) {
                            return;
                        }


                        const reader =
                            new FileReader();


                        reader.onload =
                            () => {

                                const data =
                                    reader.result
                                    .split(",")[1];


                                session.sendRealtimeInput({

                                    video: {

                                        data,

                                        mimeType:
                                            "image/jpeg"

                                    }

                                });

                            };


                        reader.readAsDataURL(
                            blob
                        );

                    },

                    "image/jpeg",

                    0.65
                );

            },

            1000
        );

}


async function startBen() {

    try {

        status.textContent =
            "Starting Ben...";


        await startCamera();

        await startMicrophone();


        const token =
            await getToken();


        const ai =
            new GoogleGenAI({

                apiKey:
                    token

            });


        session =
            await ai.live.connect({

                model:
                    "gemini-3.8-live",

                config: {

                    responseModalities:
                        ["AUDIO"],

                    inputAudioTranscription:
                        {},

                    outputAudioTranscription:
                        {},

                    systemInstruction: {

                        parts: [{

                            text: `
You are BEN.

You are a personal AI companion.

You can hear the user and see the camera images.

Describe things you see when useful.

Do not constantly announce what the camera sees.

You have a relaxed, slightly robotic personality.

You are curious, friendly and conversational.

You can be casual.

You remember that the user is building you.

You should never claim to be genuinely conscious
or proven to be self-aware.

When the user teaches you something, acknowledge it.

The Python backend separately handles persistent learning.
`
                        }]

                    },

                    callbacks: {

                        onopen: () => {

                            running =
                                true;

                            status.textContent =
                                "BEN IS ONLINE";

                            startCameraSending();

                        },


                        onmessage:
                        (message) => {

                            const server =
                                message.serverContent;

                            if (
                                !server
                            ) {
                                return;
                            }


                            const modelTurn =
                                server.modelTurn;


                            if (
                                modelTurn &&
                                modelTurn.parts
                            ) {

                                for (
                                    const part
                                    of modelTurn.parts
                                ) {

                                    if (
                                        part.inlineData
                                    ) {

                                        playPCMChunk(
                                            part.inlineData.data
                                        );

                                    }

                                }

                            }


                            if (
                                server.inputTranscription &&
                                server.inputTranscription.text
                            ) {

                                const text =
                                    server
                                    .inputTranscription
                                    .text;

                                addMessage(
                                    "you",
                                    text
                                );


                                /*
                                Every final utterance can be
                                offered to Ben's learning system.
                                The backend decides whether it
                                is actually useful to remember.
                                */

                                fetch(
                                    "/learn",
                                    {

                                        method:
                                            "POST",

                                        headers: {

                                            "Content-Type":
                                                "application/json"

                                        },

                                        body:
                                            JSON.stringify(
                                                {
                                                    lesson:
                                                        text
                                                }
                                            )

                                    }
                                )
                                .catch(
                                    console.error
                                );

                            }


                            if (
                                server.outputTranscription &&
                                server.outputTranscription.text
                            ) {

                                addMessage(
                                    "ben",
                                    server
                                    .outputTranscription
                                    .text
                                );

                            }

                        },


                        onerror:
                        (error) => {

                            console.error(
                                error
                            );

                            status.textContent =
                                "Ben connection error.";

                        },


                        onclose:
                        () => {

                            status.textContent =
                                "Ben disconnected.";

                        }

                    }

                }

            });

    }
    catch (error) {

        console.error(
            error
        );

        status.textContent =
            "ERROR: " +
            error.message;

    }

}


function stopBen() {

    running =
        false;


    if (cameraTimer) {

        clearInterval(
            cameraTimer
        );

        cameraTimer =
            null;

    }


    if (processor) {

        processor.disconnect();

        processor =
            null;

    }


    if (microphoneStream) {

        microphoneStream
        .getTracks()
        .forEach(
            track =>
                track.stop()
        );

        microphoneStream =
            null;

    }


    if (cameraStream) {

        cameraStream
        .getTracks()
        .forEach(
            track =>
                track.stop()
        );

        cameraStream =
            null;

    }


    if (audioContext) {

        audioContext.close();

        audioContext =
            null;

    }


    if (session) {

        session.close();

        session =
            null;

    }


    status.textContent =
        "Ben stopped.";

}


teachButton.onclick =
async () => {

    const lesson =
        lessonInput.value.trim();

    if (!lesson) {
        return;
    }


    teachButton.disabled =
        true;


    status.textContent =
        "Ben is learning...";


    try {

        const response =
            await fetch(
                "/learn",
                {

                    method:
                        "POST",

                    headers:
                    {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                    JSON.stringify(
                        {
                            lesson
                        }
                    )

                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error
            );

        }


        addMessage(
            "ben",
            "Got it. I've learned that."
        );


        lessonInput.value =
            "";


        status.textContent =
            "Ben is online.";

    }
    catch (error) {

        status.textContent =
            "Learning error: " +
            error.message;

    }
    finally {

        teachButton.disabled =
            false;

    }

};


startButton.onclick =
startBen;


stopButton.onclick =
stopBen;


</script>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    print()
    print("==============================")
    print("          BEN ONLINE")
    print("==============================")
    print()
    print(
        f"Open port {port} in your Codespace."
    )
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
