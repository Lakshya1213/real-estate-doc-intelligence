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


function searchQuery() {
    const query = document.getElementById("queryInput").value;
    const loader = document.getElementById("loader");
    const resultsDiv = document.getElementById("results");
    const latencyText = document.getElementById("latency");

    if (!query) {
        alert("Please enter a question.");
        return;
    }

    resultsDiv.innerHTML = "";
    loader.style.display = "block";
    latencyText.innerText = "";

    const start = performance.now();

    setTimeout(() => {
        const mockResults = [
            {
                text: "The property is located near Metro Station and City Mall.",
                pdf_name: "sale_agreement.pdf",
                page: 3
            },
            {
                text: "The total area of the land is 2400 square feet.",
                pdf_name: "property_details.pdf",
                page: 5
            },
            {
                text: "The building includes parking space for 20 vehicles.",
                pdf_name: "layout_plan.pdf",
                page: 2
            }
        ];

        loader.style.display = "none";

        const end = performance.now();
        const latency = ((end - start) / 1000).toFixed(2);
        latencyText.innerText = "Query Latency: " + latency + "s";

        mockResults.forEach(item => {
            const div = document.createElement("div");
            div.classList.add("result-item");
            div.innerHTML = `
                <p>${item.text}</p>
                <small>Source: ${item.pdf_name}, Page ${item.page}</small>
            `;
            resultsDiv.appendChild(div);
        });

    }, 1200);
}


function sendChatMessage() {
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

    // Simulate AI response
    setTimeout(() => {
        const aiMsg = document.createElement("div");
        aiMsg.classList.add("chat-message", "ai");
        aiMsg.innerText = generateMockResponse(text);
        chatBox.appendChild(aiMsg);

        chatBox.scrollTop = chatBox.scrollHeight;
    }, 800);

    chatBox.scrollTop = chatBox.scrollHeight;
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
