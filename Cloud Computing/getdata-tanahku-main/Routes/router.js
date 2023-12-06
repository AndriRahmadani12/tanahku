const express = require('express');
const router =express.Router();

const control = require('../controller/control');

router.get('/tanaman', control.getAllTanaman)
router.get('/tanaman/:id_tanaman', control.getIdTanaman)

module.exports = router;