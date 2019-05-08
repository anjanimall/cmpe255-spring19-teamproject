var myApp = angular.module("my-app", []);
myApp.controller("PopupCtrl", ['$scope', '$http', function($scope, $http){
   console.log("Controller Initialized");
   $scope.selectedText = "";
   $scope.neutralStyle = {};
   $scope.score = "";
   $scope.magnitude = "";
   $scope.harmlessWidth = {'width': 0 + "px"};
   $scope.harmlessPercentage = "";
   $scope.toxicWidth = {'width': 0 + "px"};
   $scope.toxicPercentage = "";
   let bgPage = chrome.extension.getBackgroundPage();
   let selectedText = bgPage.selectedText;
   $scope.selectedText = selectedText;
   console.log("selectedText: " + selectedText);
   if(selectedText.length > 0) {
     $http({
          url: 'https://language.googleapis.com/v1/documents:analyzeSentiment?key=YOUR_KEY',
          method: "POST",
          data: {
            "encodingType": "UTF8",
            "document": {
              "type": "PLAIN_TEXT",
              "content": selectedText
            }
          }
      })
      .then(function(response) {
        // success
        console.log("success in getting reponse from Sentiment Analysis API");
        console.log("response: " + JSON.stringify(response));
        $scope.score = response.data.documentSentiment.score;
        $scope.magnitude = response.data.documentSentiment.magnitude;
        updateSentimentAnalyzer();
      },
      function(response) { // optional
        // failed
        console.log("failure in getting response from Sentiment Analysis API");
      });
   }
   function updateSentimentAnalyzer() {
    let percentage = Math.abs($scope.score) * 100;
    if($scope.score > 0) {
      // harmless
      $scope.harmlessWidth = {'width': percentage + "%"};
      $scope.harmlessPercentage = percentage;
    }else if($scope.score < 0) {
      // toxic
      $scope.toxicWidth = {'width': percentage + "%"};
      $scope.toxicPercentage = percentage;
    }else {
      // Neutral
      $scope.harmlessWidth = {'width': "5px"};
      $scope.toxicWidth = {'width': "5px"};
      $scope.neutralStyle = {'font-weight':"bold"};
    }
   }
  }
]);
