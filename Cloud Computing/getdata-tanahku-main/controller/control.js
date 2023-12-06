const { request } = require('express');
const pool = require('../database/db');
const control ={
    
    getAllTanaman:async (req,res) => {
        try {
            const [rows,fields] = await pool.query("select * from tanaman");
            res.json({
                message: 'get all tanaman',
                data: rows
            })
        } catch (error) {
            console.log(error)
        }
    },
    getIdTanaman:async (req,res) => {
        try {
            const {id_tanaman} = req.params
            const [rows,fields] = await pool.query("select * from tanaman where id_tanaman =?", [id_tanaman]);
            res.json({
                data: rows
            })
        } catch (error) {
            
        }
    }

}

module.exports = control;