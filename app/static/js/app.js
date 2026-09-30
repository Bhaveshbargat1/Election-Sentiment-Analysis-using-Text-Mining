// Election Sentiment Mining Client-side JS

document.addEventListener("DOMContentLoaded", function () {
    // Initialize tooltips if Bootstrap tooltip is available
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    console.log("Election Social Media Sentiment Mining UI initialized.");
});
