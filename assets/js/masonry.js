$(document).ready(function() {
  // Initialize Masonry if grid exists
  var $grid = $('.grid');
  if ($grid.length && typeof $grid.masonry === 'function') {
    $grid.masonry({
      gutter: 10,
      horizontalOrder: true,
      itemSelector: '.grid-item',
    });
    
    // Layout Masonry after each image loads
    if (typeof imagesLoaded !== 'undefined') {
      $grid.imagesLoaded().progress(function() {
        $grid.masonry('layout');
      });
    }
  }
});
