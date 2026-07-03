import http from 'k6/http';

// export const options = {
//     vus: 10,
//     iterations: 10,
// };

export default function () {
    const url = 'https://shl-ai-recommender-jlw3.onrender.com/chat';

    const payload = JSON.stringify({
        messages: [
            {
                role: "user",
                content: "I am hiring a Sales Executive."
            }
        ]
    });

    const params = {
        headers: {
            "Content-Type": "application/json",
        },
    };

    const res = http.post(url, payload, params);

    console.log("Status:", res.status);
    console.log("Body:", res.body);
}