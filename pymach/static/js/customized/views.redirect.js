$(function(){
    
    $("#viewDefine").on("click", function(){
        window.location.replace("/defineData");
    });
    $("#viewAnalyze").on("click", function(){
        window.location.replace("/analyze_base");
    });
    $("#viewModels").on("click", function(){
        window.location.replace("/model_base");
    });
    $("#viewImprove").on("click", function(){
        window.location.replace("/improve_base");
    });
    // $("#viewPrediction").on("click", function(){
        // window.location.replace("/prediction");
    // });
    $("#viewMarket").on("click", function(){
        window.location.replace("/market_base");
    })
    
});
