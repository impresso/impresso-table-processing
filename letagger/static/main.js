let TAGS = [];

function setGlobals(data) {
    TAGS = data['tags']
}

$(window).on('load', function() {

    let id_tags = {};

    //Containers
    let rightContainer = $('#rightContainer');

    //Tables
    let tableLeft = $('#tableLeft tbody')[0];
    let curRowLeft;  //currently selected row
    let tableRight = $('#tableRight tbody')[0];
    let curRowRight;  //currently selected row

    //Buttons
    let btnZoomLeft = $('#btnZoomLeft');
    let btnCreate = $('#btnCreate');
    let btnSimilar = $('#btnSimilar');
    let btnSave = $('#btnSave');
    let btnExtractTags = $('#btnExtractTags');
    let btnExtractDataset = $('#btnExtractDataset');
    let btnAutoTag = $('#btnAutoTag');
    let btnZoomRight = $('#btnZoomRight');

    //Selects
    let selectLeft = $('#selectLeft');

    //Images
    let imgLeft = $('#imgLeft');
    let imgRight = $('#imgRight');

    //Modals
    let modalSave = $('#modalSave');
    let modalOperation = $('#modalOperation');

    //Functions
    function tagRow(row, tag) {
        if (typeof row !== "undefined") {
            if (row.children('td')[2].textContent === tag) {
                row.children('td')[2].textContent = "";
                id_tags[row.children('td')[1].textContent] = {"tag": "", "date": new Date().toLocaleString()};
            } else {
                row.children('td')[2].textContent = tag;
                id_tags[row.children('td')[1].textContent] = {"tag": tag, "date": new Date().toLocaleString()}
            }

            curRowLeft = curRowLeft.next('tr');
            curRowLeft.addClass('table-active').siblings().removeClass('table-active');
            fetch("/img?" + new URLSearchParams({
                id: curRowLeft.children('td')[1].textContent})).then(res => {
                return res.text();
            }).then(img => {
                imgLeft.attr("src", "data:image/jpeg;base64," + img);
                btnZoomLeft.removeClass("d-none")
            });
        }
    }

    function createButtons(tag){
        let btnLeft = document.createElement("button");
        btnLeft.innerHTML = tag;
        btnLeft.className += 'btn btn-outline-primary btn-sm';
        btnLeft.onclick = function () {
            tagRow(curRowLeft, tag);
        };
        $('#btnLeft').append(btnLeft);

        let btnRight = document.createElement("button");
        btnRight.innerHTML = tag;
        btnRight.className += 'btn btn-outline-primary btn-sm';
        btnRight.onclick = function () {
            tagRow(curRowRight, tag);
        };
        $('#btnRight').append(btnRight);
    }

    function saveTags(spinner){
        spinner.removeClass('d-none');
        fetch("/save", {
            method: "POST",
            headers: new Headers({'content-type': 'application/json'}),
            body: JSON.stringify(id_tags)
        }).then(res => {
            return res.text();
        }).then(msg => {
            $('#modalOperation .modal-body').html(msg);
            modalOperation.modal('toggle');
            spinner.addClass('d-none');
        });
        id_tags = {};
    }

    $("#btnModalSave").click(function() {
        saveTags($('#btnModalSave .spinner'));
    });

    $("#btnModalNoSave").click(function (){
        modalSave.modal('hide');
        id_tags = {};
    })

    $("#btnModalOperationOK").click(function (){
        modalOperation.modal('hide');
    })

    $("#tableLeft tbody tr").click(function () {
        curRowLeft = $(this);
        $(this).addClass('table-active').siblings().removeClass('table-active');
        rightContainer.addClass("d-none")
        $("#tableRight tbody tr").remove();
        imgRight.attr("src", "");
        var img_id = curRowLeft.children('td')[1].textContent;
        fetch("/img?" + new URLSearchParams({
            id: img_id})).then(res => {
            return res.text();
        }).then(img => {
            imgLeft.attr("src", "data:image/jpeg;base64," + img);
            btnZoomLeft.removeClass("d-none")
        });
    });

    selectLeft.change(function () {

        if (Object.keys(id_tags).length >= 1){
            modalSave.modal('toggle');
        }

        fetch("/filter?" + new URLSearchParams({
            type: this.value
        })).then(res => {
            return res.text();
        }).then(list => {
            $("#tableLeft tbody tr").remove();
            list = JSON.parse(list)
            list.forEach((x, i) => {
                let row = tableLeft.insertRow();
                let index = row.insertCell(0);
                index.innerHTML = i + 1;
                let id = row.insertCell(1);
                id.innerHTML = x["id"];
                let tag = row.insertCell(2);
                tag.innerHTML = x["tag"];
            });

            $("#tableLeft tbody tr").click(function () {
                curRowLeft = $(this);
                $(this).addClass('table-active').siblings().removeClass('table-active');
                let img_id = curRowLeft.children('td')[1].textContent;
                fetch("/img?" + new URLSearchParams({
                    id: img_id
                })).then(res => {
                    return res.text();
                }).then(img => {
                    imgLeft.attr("src", "data:image/jpeg;base64," + img);
                    btnZoomLeft.removeClass("d-none");
                });
            });
        });
    });

    btnZoomLeft.click(function () {
        if (typeof curRowLeft !== "undefined") {
            let img_id = curRowLeft.children('td')[1].textContent;
            let spinner = $('#btnZoomLeft .spinner-border');
            spinner.removeClass('d-none');
            fetch("/zoom?" + new URLSearchParams({
                id: img_id})).then(res => {
                return res.text();
            }).then(img => {
                imgLeft.attr("src", "data:image/jpeg;base64," + img);
                spinner.addClass('d-none');
            });
        }
    });

    TAGS.forEach((x, i) => {
        createButtons(x);
        selectLeft.append($('<option>', {
            value: x,
            text: x
        }));
    });

    btnCreate.click(function () {
        let tag = $('#textCreate').val();
        TAGS.push(tag);
        createButtons(tag);
        selectLeft.append($('<option>', {
            value: tag,
            text: tag
        }));
    })

    btnSimilar.click(function () {
        if (typeof curRowLeft !== "undefined") {
            rightContainer.removeClass("d-none")
            let img_id = curRowLeft.children('td')[1].textContent;
            let spinner = $('#btnSimilar .spinner-border');
            spinner.removeClass('d-none');
            fetch("/sim?" + new URLSearchParams({
                id: img_id})).then(res => {
                return res.text();
            }).then(list => {
                $("#tableRight tbody tr").remove();
                list = JSON.parse(list)
                list.forEach((x, i) => {
                    let row = tableRight.insertRow();
                    let score = row.insertCell(0);
                    score.innerHTML = x["score"].toFixed(2);
                    let id = row.insertCell(1);
                    id.innerHTML = x["id"];
                    let tag = row.insertCell(2);
                    tag.innerHTML = x["tag"];

                    if (i === 0) {
                        fetch("/img?" + new URLSearchParams({
                            id: x["id"]
                        })).then(res => {
                            return res.text();
                        }).then(img => {
                            imgRight.attr("src", "data:image/jpeg;base64," + img);
                            btnZoomRight.removeClass("d-none");
                            row.classList.add('table-active')
                        });
                    }
                });
                $("#tableRight tbody tr").click(function () {
                    curRowRight = $(this);
                    $(this).addClass('table-active').siblings().removeClass('table-active');
                    var img_id = curRowRight.children('td')[1].textContent;
                    fetch("/img?" + new URLSearchParams({
                        id: img_id})).then(res => {
                        return res.text();
                    }).then(img => {
                        imgRight.attr("src", "data:image/jpeg;base64," + img);
                        btnZoomRight.removeClass("d-none")
                    });
                });
                spinner.addClass('d-none');
            });
        }
    });

    btnSave.click(function() {
        saveTags($('#btnSave .spinner'));
    });

    btnExtractTags.click(function () {
        $(this).prop("disabled", true);
        let spinner = $('#btnExtractTags .spinner-border');
        spinner.removeClass('d-none');
        fetch("/extract?" + new URLSearchParams({
            tags: "true"})).then(res => {
            return res.text()
        }).then(msg => {
            $('#modalOperation .modal-body').html(msg);
            modalOperation.modal('toggle');
            spinner.addClass('d-none');
            $(this).prop("disabled", false);
        });
    });

    btnExtractDataset.click(function () {
        let spinner = $('#btnExtractDataset .spinner-border');
        spinner.removeClass('d-none');
        $(this).prop("disabled", true);
        fetch("/extract?" + new URLSearchParams({
            tags: "false"})).then(res => {
            return res.text()
        }).then(msg => {
            $('#modalOperation .modal-body').html(msg);
            modalOperation.modal('toggle');
            spinner.addClass('d-none');
            $(this).prop("disabled", false);
        });
    });

    btnAutoTag.click(function () {
        let spinner = $('#btnAutoTag .spinner-border');
        spinner.removeClass("d-none");
        $(this).prop("disabled", true);
        fetch("/auto?").then(res => {
            console.log(res)
            return res.text()
        }).then(msg => {
            console.log(msg)
            $('#modalOperation .modal-body').html(msg);
            modalOperation.modal('toggle');
            spinner.addClass('d-none');
            $(this).prop("disabled", false);
        })
    });

    btnZoomRight.click(function () {
        if (typeof curRowRight !== "undefined") {
            let img_id = curRowRight.children('td')[1].textContent;
            let spinner = $('#btnZoomRight .spinner-border');
            spinner.removeClass("d-none");
            fetch("/zoom?" + new URLSearchParams({
                id: img_id})).then(res => {
                return res.text();
            }).then(img => {
                imgRight.attr("src", "data:image/jpeg;base64," + img);
            });
            spinner.addClass('d-none');
        }
    });
});
