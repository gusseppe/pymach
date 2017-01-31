$(function(){

    function initFlot(){
        var data1 = [
                [1, 20],
                [2, 20],
                [3, 40],
                [4, 30],
                [5, 40],
                [6, 35],
                [7, 47]
            ],
            data2 = [
                [1, 13],
                [2, 8],
                [3, 17],
                [4, 10],
                [5, 17],
                [6, 15],
                [7, 16]
            ],
            data3 = [
                [1, 23],
                [2, 13],
                [3, 33],
                [4, 16],
                [5, 32],
                [6, 28],
                [7, 31]
            ],
            $chart = $("#flot-main"),
            $tooltip = $('#flot-main-tooltip');

        function _initChart(){
            var plot = $.plotAnimator($chart, [{
                label: "Traffic",
                data: data3,
                lines: {
                    fill: .3,
                    lineWidth: 0
                },
                color:['#ccc']
            },{
                label: "Traffic",
                data: data2,
                lines: {
                    fill: 0.6,
                    lineWidth: 0
                },
                color:['#F7653F']
            },{
                label: "Traffic",
                data: data1,
                animator: {steps: 60, duration: 1000, start:0},
                lines: {lineWidth:2},
                shadowSize:0,
                color: '#F7553F'
            }],{
                xaxis: {
                    tickLength: 0,
                    tickDecimals: 0,
                    min:2,
                    font :{
                        lineHeight: 13,
                        weight: "bold",
                        color: Sing.colors['gray-semi-light']
                    }
                },
                yaxis: {
                    tickDecimals: 0,
                    tickColor: "#f3f3f3",
                    font :{
                        lineHeight: 13,
                        weight: "bold",
                        color: Sing.colors['gray-semi-light']
                    }
                },
                grid: {
                    backgroundColor: { colors: [ "#fff", "#fff" ] },
                    borderWidth:1,
                    borderColor:"#f0f0f0",
                    margin:0,
                    minBorderMargin:0,
                    labelMargin:20,
                    hoverable: true,
                    clickable: true,
                    mouseActiveRadius:6
                },
                legend: false
            });

            $chart.on("plothover", function (event, pos, item) {
                if (item) {
                    var x = item.datapoint[0].toFixed(2),
                        y = item.datapoint[1].toFixed(2);

                    $tooltip.html(item.series.label + " at " + x + ": " + y)
                        .css({
                            top: item.pageY + 5 - window.scrollY,
                            left: item.pageX + 5 - window.scrollX
                        })
                        .fadeIn(200);
                } else {
                    $tooltip.hide();
                }
            });
        }

        _initChart();

        SingApp.onResize(_initChart);
    }

    function initRickshaw(){
        "use strict";

        var seriesData = [ [], [] ];
        var random = new Rickshaw.Fixtures.RandomData(30);

        for (var i = 0; i < 30; i++) {
            random.addData(seriesData);
        }

        var graph = new Rickshaw.Graph( {
            element: document.getElementById("rickshaw"),
            height: 130,
            renderer: 'area',
            series: [
                {
                    color: '#96E593',
                    data: seriesData[0],
                    name: 'Uploads'
                }, {
                    color: '#ecfaec',
                    data: seriesData[1],
                    name: 'Downloads'
                }
            ]
        } );

        function onResize(){
            var $chart = $('#rickshaw');
            graph.configure({
                width: $chart.width(),
                height: 130
            });
            graph.render();

            $chart.find('svg').css({height: '130px'});
        }

        SingApp.onResize(onResize);
        onResize();


        var hoverDetail = new Rickshaw.Graph.HoverDetail( {
            graph: graph,
            xFormatter: function(x) {
                return new Date(x * 1000).toString();
            }
        } );

        setInterval( function() {
            random.removeData(seriesData);
            random.addData(seriesData);
            graph.update();

        }, 1000 );
    }

    function initSparkline1(){
        function _initChart(){
            $('#sparkline1').sparkline([2,4,6,2,7,5,3,7,8,3,6],{
                width: '100%',
                fillColor: '#ddd',
                height: '100px',
                lineColor: 'transparent',
                spotColor: '#c0d0f0',
                minSpotColor: null,
                maxSpotColor: null,
                highlightSpotColor: '#ddd',
                highlightLineColor: '#ddd'
            }).sparkline([5,3,7,8,3,6,2,4,6,2,7],{
                composite: true,
                lineColor: 'transparent',
                spotColor: '#c0d0f0',
                fillColor: 'rgba(192, 208, 240, 0.76)',
                minSpotColor: null,
                maxSpotColor: null,
                highlightSpotColor: '#ddd',
                highlightLineColor: '#ddd'
            })
        }

        _initChart();

        SingApp.onResize(_initChart);
    }

    function initSparkline2(){
        $('#sparkline2').sparkline([2,4,6],{
            type: 'pie',
            width: '100px',
            height: '100px',
            sliceColors: ['#F5CB7B', '#FAEEE5', '#f0f0f0']
        });
    }

    /* Inspired by Lee Byron's test data generator. */
    function _stream_layers(n, m, o) {
        if (arguments.length < 3) o = 0;
        function bump(a) {
            var x = 1 / (.1 + Math.random()),
                y = 2 * Math.random() - .5,
                z = 10 / (.1 + Math.random());
            for (var i = 0; i < m; i++) {
                var w = (i / m - y) * z;
                a[i] += x * Math.exp(-w * w);
            }
        }
        return d3.range(n).map(function() {
            var a = [], i;
            for (i = 0; i < m; i++) a[i] = o + o * Math.random();
            for (i = 0; i < 5; i++) bump(a);
            return a.map(function(d, i) {
                return {x: i, y: Math.max(0, d)};
            });
        });
    }

    function testData(stream_names, pointsCount) {
        var now = new Date().getTime(),
            day = 1000 * 60 * 60 * 24, //milliseconds
            daysAgoCount = 60,
            daysAgo = daysAgoCount * day,
            daysAgoDate = now - daysAgo,
            pointsCount = pointsCount || 45, //less for better performance
            daysPerPoint = daysAgoCount / pointsCount;
        return _stream_layers(stream_names.length, pointsCount, .1).map(function(data, i) {
            return {
                key: stream_names[i],
                values: data.map(function(d,j){
                    return {
                        x: daysAgoDate + d.x * day * daysPerPoint,
                        y: Math.floor(d.y * 100) //just a coefficient,
                    }
                })
            };
        });
    }

    function initNvd31(){

        nv.addGraph(function() {
            var chart = nv.models.lineChart()
                .useInteractiveGuideline(true)
                .margin({left: 28, bottom: 30, right: 0})
                .color(['#82DFD6', '#ddd']);

            chart.xAxis
                .showMaxMin(false)
                .tickFormat(function(d) { return d3.time.format('%b %d')(new Date(d)) });

            chart.yAxis
                .showMaxMin(false)
                .tickFormat(d3.format(',f'));

            d3.select('#nvd31 svg')
                .style('height', '300px')
                .datum(testData(['Search', 'Referral'], 50).map(function(el, i){
                    el.area = true;
                    return el;
                }))
                .transition().duration(500)
                .call(chart)
            ;


            SingApp.onResize(chart.update);

            return chart;
        });
    }

    function initNvd32(){

        nv.addGraph(function() {
            var chart = nv.models.multiBarChart()
                .margin({left: 28, bottom: 30, right: 0})
                .color(['#F7653F', '#ddd']);

            chart.xAxis
                .showMaxMin(false)
                .tickFormat(function(d) { return d3.time.format('%b %d')(new Date(d)) });

            chart.yAxis
                .showMaxMin(false)
                .tickFormat(d3.format(',f'));

//            chart.controls.margin({left: 0});

            d3.select('#nvd32 svg')
                .style('height', '300px')
                .datum(testData(['Uploads', 'Downloads'], 10).map(function(el, i){
                    el.area = true;
                    return el;
                }))
                .transition().duration(500)
                .call(chart)
            ;


            SingApp.onResize(chart.update);

            return chart;
        });
    }

    function initMorris1(){
        $('#morris1').css({height: '343px'}); //safari svg height fix
        Morris.Line({
            element: 'morris1',
            resize: true,
            data: [
                { y: '2006', a: 100, b: 90 },
                { y: '2007', a: 75,  b: 65 },
                { y: '2008', a: 50,  b: 40 },
                { y: '2009', a: 75,  b: 65 },
                { y: '2010', a: 50,  b: 40 },
                { y: '2011', a: 75,  b: 65 },
                { y: '2012', a: 100, b: 90 }
            ],
            xkey: 'y',
            ykeys: ['a', 'b'],
            labels: ['Series A', 'Series B'],
            lineColors: ['#88C4EE', '#ccc']
        });
    }

    function initMorris2(){
        $('#morris2').css({height: '343px'}); //safari svg height fix
        Morris.Area({
            element: 'morris2',
            resize: true,
            data: [
                { y: '2006', a: 100, b: 90 },
                { y: '2007', a: 75,  b: 65 },
                { y: '2008', a: 50,  b: 40 },
                { y: '2009', a: 75,  b: 65 },
                { y: '2010', a: 50,  b: 40 },
                { y: '2011', a: 75,  b: 65 },
                { y: '2012', a: 100, b: 90 }
            ],
            xkey: 'y',
            ykeys: ['a', 'b'],
            labels: ['Series A', 'Series B'],
            lineColors: ['#80DE78', '#9EEE9B'],
            lineWidth: 0
        });
    }

    function initMorris3(){
        $('#morris3').css({height: 180});
        Morris.Donut({
            element: 'morris3',
            data: [
                {label: "Download Sales", value: 12},
                {label: "In-Store Sales", value: 30},
                {label: "Mail-Order Sales", value: 20}
            ],
            colors: ['#F7653F', '#F8C0A2', '#e6e6e6']
        });

    }

    function initEasyPie(){
        $('#easy-pie1').easyPieChart({
            barColor: '#5dc4bf',
            trackColor: '#ddd',
            scaleColor: false,
            lineWidth: 10,
            size: 120
        });
    }

    function initFlotBar(){
        var bar_customised_1 = [[1388534400000, 120], [1391212800000, 70],  [1393632000000, 100], [1396310400000, 60], [1398902400000, 35]];
        var bar_customised_2 = [[1388534400000, 90], [1391212800000, 60], [1393632000000, 30], [1396310400000, 73], [1398902400000, 30]];
        var bar_customised_3 = [[1388534400000, 80], [1391212800000, 40], [1393632000000, 47], [1396310400000, 22], [1398902400000, 24]];

        var data = [
            {
                label: "Apple",
                data: bar_customised_1,
                bars: {
                    show: true,
                    barWidth: 12*24*60*60*300,
                    fill: true,
                    lineWidth:0,
                    order: 1
                }
            },
            {
                label: "Google",
                data: bar_customised_2,
                bars: {
                    show: true,
                    barWidth: 12*24*60*60*300,
                    fill: true,
                    lineWidth: 0,
                    order: 2
                }
            },
            {
                label: "Facebook",
                data: bar_customised_3,
                bars: {
                    show: true,
                    barWidth: 12*24*60*60*300,
                    fill: true,
                    lineWidth: 0,
                    order: 3
                }
            }

        ];

        function _initChart(){
            $.plot($("#flot-bar"), data, {
                series: {
                    bars: {
                        show: true,
                        barWidth: 12*24*60*60*350,
                        lineWidth: 0,
                        order: 1,
                        fillColor: {
                            colors: [{
                                opacity: 1
                            }, {
                                opacity: 0.7
                            }]
                        }
                    }
                },
                xaxis: {
                    mode: "time",
                    min: 1387497600000,
                    max: 1400112000000,
                    tickLength: 0,
                    tickSize: [1, "month"],
                    axisLabel: 'Month',
                    axisLabelUseCanvas: true,
                    axisLabelFontSizePixels: 13,
                    axisLabelPadding: 15
                },
                yaxis: {
                    axisLabel: 'Value',
                    axisLabelUseCanvas: true,
                    axisLabelFontSizePixels: 13,
                    axisLabelPadding: 5
                },
                grid: {
                    hoverable: true,
                    borderWidth: 0
                },
                legend: {
                    backgroundColor: "transparent",
                    labelBoxBorderColor: "none"
                },
                colors: ["#64bd63", "#f0b518", "#F7653F"]
            });
        }

        _initChart();

        SingApp.onResize(_initChart);


    }

    function pageLoad(){
        $('.widget').widgster();
        $('.sparkline').each(function(){
            $(this).sparkline('html', $(this).data());
        });

        initFlot();
        initRickshaw();
        initSparkline1();
        initSparkline2();
        initNvd31();
        initNvd32();
        initMorris1();
        initMorris2();
        initMorris3();
        initEasyPie();
        initFlotBar();
    }
    pageLoad();
    SingApp.onPageLoad(pageLoad);
});