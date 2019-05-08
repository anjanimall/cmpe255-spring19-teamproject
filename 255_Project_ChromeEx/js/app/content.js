console.log("Content js script loaded.");

window.addEventListener("mouseup", wordSelected);
function wordSelected() {
  let selectedText = window.getSelection().toString();
  if(selectedText.length > 0) {
      console.log("selectedText: " + selectedText);
      chrome.runtime.sendMessage(selectedText);
  }
}
