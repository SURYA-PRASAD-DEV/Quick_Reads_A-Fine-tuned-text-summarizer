function summarizeText() {
    let text = document.getElementById("inputText").value;
    let numLines = document.getElementById("numLines").value;
    let language = document.getElementById("language").value;

    if (text.trim() === "") {
        alert("Please enter some text!");
        return;
    }

    fetch("/summarize", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text, num_lines: numLines, language: language })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("summaryText").innerText = data.summary;
        let audioPlayer = document.getElementById("audioPlayer");
        audioPlayer.src = data.audio_url;
        audioPlayer.style.display = "block";
        document.getElementById("playAudio").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
}

function playAudio() {
    document.getElementById("audioPlayer").play();
}
