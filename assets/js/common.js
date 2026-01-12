$(document).ready(function() {
    // Handle abstract toggle
    $('a.abstract').on('click', function(e) {
        e.preventDefault();
        $(this).parent().parent().find(".abstract.hidden").toggleClass('open');
    });
    
    // Handle bibtex toggle
    $('a.bibtex').on('click', function(e) {
        e.preventDefault();
        $(this).parent().parent().find(".bibtex.hidden").toggleClass('open');
    });
    
    // Remove waves effect classes if present
    $('a').removeClass('waves-effect waves-light');
});
