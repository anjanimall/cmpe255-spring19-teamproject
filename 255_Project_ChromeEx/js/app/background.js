console.log("background script loaded.");
chrome.runtime.onMessage.addListener(receiver);
var selectedText = "";
function receiver(message, sender, sendResponse) {
  console.log("message Received: " + message);
  selectedText = message;
}
