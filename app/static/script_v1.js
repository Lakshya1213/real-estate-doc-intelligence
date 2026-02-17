console.log("Loded Script_v1.js")

async function uploaddocument() {
    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("uploadStatus");

    if (fileInput.files.length === 0) {
        status.innerText = "Please select a file first.";
        status.style.color = "red";
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    status.innerText = "Uploading & processing...";
    status.style.color = "blue";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            status.innerText = `Success! ${data.filename} chunks created.`;
            status.style.color = "green";
        } else {
            status.innerText = data.detail || "Upload failed.";
            status.style.color = "red";
        }

    } catch (error) {
        status.innerText = "Server error.";
        status.style.color = "red";
    }
}




async function sendChatMessage() {
    const input = document.getElementById("chatInput");
    const text = input.value.trim();
    const chatBox = document.getElementById("chatBox");

    if (text === "") return;

    const userMsg = document.createElement("div");
    userMsg.classList.add("chat-message", "user");
    userMsg.innerText = text;
    chatBox.appendChild(userMsg);

    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: text,
                top_k: 3
            })
        });

        const data = await response.json();
        console.log(data);

        const aiMsg = document.createElement("div");
        aiMsg.classList.add("chat-message", "ai");
        aiMsg.innerText = data.answer;

        chatBox.appendChild(aiMsg);
        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (error) {
        console.error("Frontend error:", error);
    }
}

// Allow Enter key
document.getElementById("chatInput").addEventListener("keypress", function(e) {
    if (e.key === "Enter") {
        sendChatMessage();
    }
});

