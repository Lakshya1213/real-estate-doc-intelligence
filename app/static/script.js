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


// async function searchQuery() {
//     const query = document.getElementById("chatInput").value;
//     const resultsDiv = document.getElementById("results");
//     const loader = document.getElementById("loader");

//     if (!query) {
//         alert("Please enter a question.");
//         return;
//     }

//     resultsDiv.innerHTML = "";
//     loader.style.display = "block";

//     const start = performance.now();

//     try {
//         const response = await fetch("/search", {
//             method: "POST",
//             headers: {
//                 "Content-Type": "application/json"
//             },
//             body: JSON.stringify({
//                 query: query,
//                 top_k: 3
//             })
//         });

//         const data = await response.json();

//         loader.style.display = "none";

//         const end = performance.now();
//         const latency = ((end - start) / 1000).toFixed(2);

//         document.getElementById("latency").innerText =
//             "Query Latency: " + latency + "s";

//         data.results.forEach(item => {
//             const div = document.createElement("div");
//             div.classList.add("result-item");
//             div.innerHTML = `
//                 <p>${item.text}</p>
//                 <small>Score: ${item.score?.toFixed(4)}</small>
//             `;
//             resultsDiv.appendChild(div);
//         });

//     } catch (error) {
//         console.error("Error:", error);
//         loader.style.display = "none";
//         alert("Error searching documents");
//     }
// }


async function sendChatMessage() {
    const input = document.getElementById("chatInput");
    const text = input.value.trim();
    const chatBox = document.getElementById("chatBox");

    if (text === "") return;

    // Add user message
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

        // Combine retrieved chunks into single answer
        let combinedText = data.results
            .map(r => r.text)
            .join("\n\n");

        const aiMsg = document.createElement("div");
        aiMsg.classList.add("chat-message", "ai");
        aiMsg.innerText = combinedText;

        chatBox.appendChild(aiMsg);
        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (error) {
        console.error(error);
    }
}


// Allow Enter key
document.getElementById("chatInput").addEventListener("keypress", function(e) {
    if (e.key === "Enter") {
        sendChatMessage();
    }
});

function generateMockResponse(question) {
    const responses = [
        "The total land area mentioned is 2400 sq ft.",
        "The property is located near Metro Station.",
        "Parking is available for 20 vehicles.",
        "The built-up area is 1800 sq ft.",
        "The property falls under residential zoning."
    ];

    return responses[Math.floor(Math.random() * responses.length)];
}
