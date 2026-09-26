const form = document.getElementById('prediction-form');

if (form) {
    const result = document.getElementById('result');
    const resultText = document.getElementById('prediction-text');
    const button = document.getElementById('predict-button');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!form.reportValidity()) return;

        const data = Object.fromEntries(new FormData(form).entries());
        button.disabled = true;
        button.querySelector('span:first-child').textContent = 'Generating estimate…';
        result.className = 'result-content';
        resultText.textContent = 'The model is processing the clinical measurements.';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const payload = await response.json();
            if (!response.ok || payload.error) throw new Error(payload.error || 'The estimate could not be generated. Please check the entered values.');

            const isHigher = payload.prediction_text.toLowerCase().includes('higher');
            result.classList.add(isHigher ? 'warning' : 'success');
            const scoreDetails = Number.isFinite(payload.model_score_percent)
                ? ` Model score: ${payload.model_score_percent}% (demo flag threshold: ${payload.screening_threshold_percent}%).`
                : '';
            resultText.textContent = `${payload.prediction_text}${scoreDetails}`;
        } catch (error) {
            result.classList.add('error');
            resultText.textContent = error.message || 'A connection issue prevented the estimate. Please try again.';
        } finally {
            button.disabled = false;
            button.querySelector('span:first-child').textContent = 'Generate screening estimate';
        }
    });
}
