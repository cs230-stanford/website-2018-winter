function changeClass() {
    element = document.getElementById("trigger");
    if (element.className == "trigger"){
        element.className = "show";
    }
    else {
        element.className = "trigger";
    }
}

function hideClass() {
    element = document.getElementById("trigger");
    if (element.className == "show") {
        element.className = "trigger";
    }
}