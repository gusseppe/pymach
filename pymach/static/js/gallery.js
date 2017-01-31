$(function(){
    function pageLoad(){
        var DEMO = (function( $ ) {
            'use strict';

            var $grid = $('#grid'),
                $filterOptions = $('.js-filter-options > .filter'),
                $sizer = $grid.find('.js-shuffle-sizer'),

                init = function() {
                    setTimeout(function() {
                        $grid.shuffle( 'shuffle', 'all' );
                    }, 0);

                    setupFilters();
                    setupSorting();
                    setupSearching();

                    SingApp.onResize(function(){
                        $grid.shuffle('update');
                    });

                    // instantiate the plugin
                    $grid.shuffle({
                        itemSelector: '.gallery-item',
                        sizer: $sizer
                    });


                },

            // Set up button clicks
                setupFilters = function() {
                    var $btns = $filterOptions;
                    $btns.on('click', function() {
                        var $this = $(this),
                            group = $this.data('group');

                        // Hide current label, show current label in title
                        $('.js-filter-options .active').removeClass('active');

                        $this.addClass('active');

                        // Filter elements
                        $grid.shuffle( 'shuffle', group );
                    });

                    $btns = null;
                },

                setupSorting = function() {
                    // Sorting options
                    $('.js-sort-options > .sort').on('click', function() {
                        var order = $(this).data('sort-order'),
                            opts = {
                            reverse: order === 'desc',
                            by: function($el) {
                                return $el.data('title').toLowerCase();
                            }

                        };
                        $('.js-sort-options .active').removeClass('active');

                        $(this).addClass('active');
                        $grid.shuffle('sort', opts);
                    });
                },

                setupSearching = function() {
                    // Advanced filtering
                    $('.js-shuffle-search').on('keyup change', function() {
                        var val = this.value.toLowerCase();
                        $grid.shuffle('shuffle', function($el, shuffle) {

                            // Only search elements in the current group
                            if (shuffle.group !== 'all' && $.inArray(shuffle.group, $el.data('groups')) === -1) {
                                return false;
                            }

                            var text = $.trim( $el.find('.picture-item__title').text() ).toLowerCase();
                            return text.indexOf(val) !== -1;
                        });
                    });
                };

            return {
                init: init
            };
        }( jQuery ));

        DEMO.init();

        $('#grid').magnificPopup({
            delegate: '.img-thumbnail > a', // child items selector, by clicking on it popup will open
            type: 'image',
            gallery: {
                enabled: true
            }
        });
    }
    pageLoad();
    SingApp.onPageLoad(pageLoad);
});