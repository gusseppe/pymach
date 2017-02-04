$(function(){
    
    $("#viewDefine").on("click", function(){
        window.location.replace("/defineData");
    });
    $("#viewAnalyze").on("click", function(){
        window.location.replace("/analyzeData");
    });
    $("#viewModels").on("click", function(){
        window.location.replace("/models");
    });
    $("#viewPrediction").on("click", function(){
        window.location.replace("/prediction");
    });
    
});
