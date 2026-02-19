console.log("Loded Script_v1.js")

async function uploaddocument() {
    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("uploadStatus");
    console.log(fileInput.files);

    if (fileInput.files.length === 0) {
        status.innerText = "Please select at least one file.";
        status.style.color = "red";
        return;
    }

    status.innerText = "Uploading & processing files...";
    status.style.color = "blue";

    try {
        for (let i = 0; i < fileInput.files.length; i++) {

            const formData = new FormData();
            formData.append("file", fileInput.files[i]);

            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                status.innerText = data.detail || "Upload failed.";
                status.style.color = "red";
                return;
            }
        }

        status.innerText = "All files uploaded successfully!";
        status.style.color = "green";

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
                top_k: 10
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
